from environment.parameters import *
from utils.recorder import DataRecorder

import omni
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim
from omni.isaac.franka import Franka
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import set_stage_units, set_stage_up_axis, is_stage_loading
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.physx.scripts.utils import setStaticCollider
from omni.kit.material.library import get_material_prim_path
from omni.isaac.dynamic_control import _dynamic_control
from omni.physx.scripts import physicsUtils

import pxr
from pxr import UsdPhysics, Gf, PhysxSchema, UsdShade
import time
from omni.isaac.synthetic_utils import SyntheticDataHelper
from abc import ABC
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.stage import save_stage
from pxr import Usd, UsdGeom

#[디버깅]===============
from PIL import Image
import os
import time

def save_robot_as_usd(robot: Franka, file_path: str):
    """
    Save the given robot as a .usd file.
    
    Args:
        robot (Franka): The robot instance to save.
        file_path (str): The file path to save the .usd file.
    """
    # Ensure the robot is initialized and valid
    # if not robot.is_initialized:
    #     robot.initialize()
    
    # Save the stage to a .usd file
    save_stage(file_path)
    print(f"Robot saved as {file_path}")

class BaseTask(ABC):
    material_library = {}
    viewport_handles = []
    
    def __init__(self, num_stages, horizon, stage_properties, cfg) -> None:
        self.cfg = cfg
        self.num_stages = num_stages
        self.horizon = horizon
        self.stage_properties: StageProperties = stage_properties
        self.timeline = omni.timeline.get_timeline_interface()
        self.kit = omni.kit.app.get_app()
        self.only_check = cfg.only_check
        self.objects_list = []
        self.recorder = None

        #[디버깅]env에서 현재 데이터 위치를 로그에 넘겨주기 위함
        self.current_data_name = None
    
    def success(self):
        if hasattr(self, "checker") and self.checker and self.checker.success:
            return True
        
        return False

    def set_up_task(self):
        raise NotImplementedError
    
    def remove_objects(self):
        for prim in self.objects_list:
            delete_prim(prim.GetPath().pathString)
        if is_prim_path_valid('/World'):
            delete_prim('/World')
        if is_prim_path_valid('/Looks'):
            delete_prim('/Looks')
        if is_prim_path_valid('/lula'):
            delete_prim('/lula')
        self.objects_list = []
        self._wait_for_loading()

    def stop(self):
        if self.recorder is not None and self.recorder.record:
            self.recorder.save_buffer(self.success())
            self.recorder = None
        self.timeline.stop()
        self._wait_for_loading()
        self.remove_objects()

    def reset(self,
              robot_parameters = None,
              scene_parameters = None,
              sensor_resolution = (128, 128),
              sensor_types = ["rgb", "depthLinear", "camera", "semanticSegmentation"],
        ):
        self.timeline.stop()
        self.checker = None
        self.kit.update()

        self.stage = omni.usd.get_context().get_stage()
        self.sensor_resolution = sensor_resolution
        self.sensor_types = sensor_types
        
        simulation_context = SimulationContext.instance()

        if robot_parameters is not None:
            self.robot_parameters = robot_parameters

        if scene_parameters is not None:
            self.scene_parameters = scene_parameters
            self.num_envs = len(scene_parameters)

        self.clear()
        self._wait_for_loading()
        self._define_stage_properties()

        if self.use_gpu_physics:
            simulation_context.get_physics_context().enable_gpu_dynamics(self.use_gpu_physics)
            simulation_context.get_physics_context().set_broadphase_type("GPU")
        
        self._load_scene()
        # self.robot = self._load_robot()
        self.set_up_task()
        self.robot = self._load_robot() # 코드 아래로 이동
        self._wait_for_loading()
        self.timeline.play()

        self.kit.update()

        def initialize(robot):
            robot.initialize()
            robot.set_joint_positions(robot._articulation_view._default_joints_state.positions)
            robot.set_joint_velocities(robot._articulation_view._default_joints_state.velocities)
            robot.set_joint_efforts(robot._articulation_view._default_joints_state.efforts)
            add_update_semantics(get_prim_at_path(robot.prim_path), "Robot")
           
            # this is important
            robot.disable_gravity()
            self.kit.update()

        initialize(self.robot)
        mesh_prim_paths = []
        # stage = self.robot
        robot_prim_path = self.robot.prim_path  # The root path of your robot
        robot_prim = self.stage.GetPrimAtPath(robot_prim_path)
        if robot_prim.IsA(UsdGeom.Mesh):
            mesh_prim_paths.append(prim.GetPath().pathString)

        if simulation_context is not None:
            for _ in range(60):
                simulation_context.step(render=False)
     
            self.checker.initialization_step()

        # settle checker (we use checker to initialze articulation body states)
        if simulation_context is not None:
            for _ in range(10):
                simulation_context.step(render=False)
        
        self.time_step = 0
        ########## setup controller
        self.gripper_controller = self.robot.gripper
        self.c_controller = RMPFlowController(name="cspace_controller", robot_articulation=self.robot, physics_dt=1/120.0)

        if self.cfg.record:
            self.register_recorder()

        return self.render()

    def step(self):
        raise NotImplementedError

    def _define_stage_properties(self):
        set_stage_up_axis(self.stage_properties.scene_up_axis)
        set_stage_units(self.stage_properties.scene_stage_unit)
        self._set_up_physics_secne()
        
        skylight_path = '/skylight'
        add_reference_to_stage(self.stage_properties.light_usd_path, skylight_path)

    def _set_up_physics_secne(self):
        # reference : https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_transfer_policy.html
        physicsScenePath = "/physicsScene"
        scene = UsdPhysics.Scene.Get(self.stage, physicsScenePath)
        if not scene:
            scene = UsdPhysics.Scene.Define(self.stage, physicsScenePath)
        
        gravityDirection = self.stage_properties.gravity_direction
        self._gravityDirection = Gf.Vec3f(gravityDirection[0], gravityDirection[1],  gravityDirection[2])

        scene.CreateGravityDirectionAttr().Set(self._gravityDirection)

        self._gravityMagnitude = self.stage_properties.gravity_magnitude
        scene.CreateGravityMagnitudeAttr().Set(self._gravityMagnitude)
        
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physxSceneAPI.CreateEnableCCDAttr().Set(True)
        physxSceneAPI.GetTimeStepsPerSecondAttr().Set(120)
        physxSceneAPI.CreateEnableGPUDynamicsAttr().Set(self.use_gpu_physics)
        physxSceneAPI.CreateEnableEnhancedDeterminismAttr().Set(True)
        physxSceneAPI.CreateEnableStabilizationAttr().Set(True)

        physxSceneAPI.GetGpuMaxRigidContactCountAttr().Set(524288)
        physxSceneAPI.GetGpuMaxRigidPatchCountAttr().Set(81920)
        physxSceneAPI.GetGpuFoundLostPairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuFoundLostAggregatePairsCapacityAttr().Set(262144)
        physxSceneAPI.GetGpuTotalAggregatePairsCapacityAttr().Set(8192)
        physxSceneAPI.GetGpuMaxSoftBodyContactsAttr().Set(1048576)
        physxSceneAPI.GetGpuMaxParticleContactsAttr().Set(1048576)
        # physxSceneAPI.GetGpuHeapCapacityAttr().Set(67108864)
        
    def render(self):
        if not self._sensor_initialized:
            return None
        
        sensor_types = [self.sensor_types] * self.num_envs * len(self.camera_paths)
        verify_sensor_inits = [False] * self.num_envs * len(self.camera_paths)
        wait_times =  [ 0.0 ] * self.num_envs * len(self.camera_paths)

        simulation_context = SimulationContext.instance()
        simulation_context.render()

        time.sleep(0.05)
        simulation_context.render()
        simulation_context.render()
        
        gts = list(map(SyntheticDataHelper.get_groundtruth, self.sd_helpers, sensor_types,
                       self.viewport_windows, verify_sensor_inits, wait_times))
        
        gts = { 'images': gts, 'semantic_id': self.sd_helpers[0].get_semantic_id_map() }
        
        return gts

    def clear(self):
        from pxr import Sdf, Usd
        index = 0
        house_prim_path = f"/World_{index}/house"
        prim_path = Sdf.Path(house_prim_path)
        prim: Usd.Prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            delete_prim(house_prim_path)
        
        # delete_prim('/physicsScene')
    
    def _load_scene(self):
        index = 0
        house_prim_path = f"/World_{index}/house"
        print("house usd path: ", self.scene_parameters[index].usd_path)
        import pdb
        # pdb.set_trace()
        # while True:
        
        house_prim = add_reference_to_stage(self.scene_parameters[index].usd_path, house_prim_path)
        self._wait_for_loading()
        furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
        room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")
          
        house_prim = XFormPrim(house_prim_path)
        # print(euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )
        house_prim.set_local_pose(np.array([0,0,0]) )
        # house_prim.set_local_pose(np.array([0,0,0]),  euler_angles_to_quat(np.array([np.pi/2, 0, 0])) )

        import pdb
        # pdb.set_trace()

        if self.only_check is False: # 추가 변경사항
            furniture_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].furniture_path}")
            #TODO 
            # somehow setting this is convexhull sometimes will lead to crash in omniverse
            setStaticCollider(furniture_prim, approximationShape=CONVEXHULL)
        
            self._wait_for_loading()

            room_struct_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].wall_path}")

            #TODO 
            # somehow setting this to convexHull will lead to some bug need to modify meshes later
            setStaticCollider(room_struct_prim, approximationShape="none")

            floor_prim = self.stage.GetPrimAtPath(f"{house_prim_path}/{self.scene_parameters[index].floor_path}")
            self._set_ground_plane(index)
            
            wall_material_url = self.scene_parameters[index].wall_material_url
            floor_material_url = self.scene_parameters[index].floor_material_url
            
            if wall_material_url and floor_material_url:
                #TODO
                # this needs some refactor 
                wall_mtl_name = wall_material_url.split("/")[-1][:-4]
                floor_mtl_name = floor_material_url.split("/")[-1][:-4]
                if wall_mtl_name not in BaseTask.material_library:
                    _, wall_material_prim_path = get_material_prim_path(wall_mtl_name)
                    BaseTask.material_library[wall_mtl_name] = wall_material_prim_path
                else:
                    wall_material_prim_path = BaseTask.material_library[wall_mtl_name]
                
                if floor_mtl_name not in BaseTask.material_library:
                    _, floor_material_prim_path = get_material_prim_path(floor_mtl_name)
                    BaseTask.material_library[floor_mtl_name] = floor_material_prim_path
                else:
                    floor_material_prim_path = BaseTask.material_library[floor_mtl_name]
                
                # print("floor_material_url: ", floor_material_url)
                if floor_material_prim_path:
                    # self._assets_root_path = get_assets_root_path()
                    # print("load floor material")
                    omni.kit.commands.execute(
                        "CreateMdlMaterialPrim",
                        mtl_url=floor_material_url,
                        mtl_name=floor_mtl_name,
                        mtl_path=floor_material_prim_path,
                        select_new_prim=False,
                    )
                    self._wait_for_loading()
                    # print("created floor material")
                    omni.kit.commands.execute(
                        "BindMaterial",
                        prim_path=floor_prim.GetPath(),
                        material_path=floor_material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants
                    )
                    self._wait_for_loading()
                    # print("load floor material done")
                
                # print("wall_material_url: ", wall_material_url)
                if wall_material_prim_path:
                    # print("load wall material")
                    omni.kit.commands.execute(
                        "CreateMdlMaterialPrim",
                        mtl_url=wall_material_url,
                        mtl_name=wall_mtl_name,
                        mtl_path=wall_material_prim_path,
                        select_new_prim=False,
                    )
                    
                    self._wait_for_loading()
                    # print("created wall material")

                    omni.kit.commands.execute(
                        "BindMaterial",
                        prim_path=room_struct_prim.GetPath(),
                        material_path=wall_material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants
                    )
                    
                    self._wait_for_loading()
                    # print("load wall material done")
        else:
            self._set_ground_plane(index)
            
        self._wait_for_loading()

    def _set_ground_plane(self, index):
        ground_plane_path = f"/World_{index}/house/groundPlane"
        physicsUtils.add_ground_plane(self.stage,  ground_plane_path, "Y", 5000.0, 
            pxr.Gf.Vec3f(0.0, 0.0, 0.0), pxr.Gf.Vec3f(0.2))
        ground_prim = self.stage.GetPrimAtPath(ground_plane_path)
        #if self.is_loading_scene:
        ground_prim.GetAttribute('visibility').Set('invisible')

    def _load_robot(self):
        # using one environment for now
        index = 0
        prim_path = f"/World_{index}/franka"

        position = self.robot_parameters[index].robot_position 
        rotation = self.robot_parameters[index].robot_orientation_quat
        
        # position, rotation = self._y_up_to_z_up(position=position, rotation=rotation)
        # import pdb
        # pdb.set_trace()
        robot = Franka(
                prim_path = prim_path, name = f"my_frankabot{index}",
                usd_path = self.robot_parameters[index].usd_path,
                orientation = rotation,
                position = position,
                end_effector_prim_name = 'panda_rightfinger',
                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
            )
        
        add_update_semantics(get_prim_at_path(prim_path), "Robot")
        self._wait_for_loading()
        self._set_sensors()
     
        return robot
    
    def _set_sensors(self):
        self._register_camera_path()    # Register camera path to use in USD scene with camera_paths
        BaseTask.sd_helpers = [] 
        BaseTask.viewport_windows = []

        from omni.usd import get_context
        stage = get_context().get_stage()   # Omniverse simulation scene called, object manipulation is possible

        #============ 디버깅 ============#
        # self._log_focal_lengths_all(stage)

        if len(BaseTask.viewport_handles) == 0: # Create Viewport, Register Camera
            for idx, camera_path in enumerate(self.camera_paths):
                print("camera_path: ", camera_path)
                viewport_handle = omni.kit.viewport_legacy.get_viewport_interface().create_instance()
                viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
                viewport_window.set_active_camera(camera_path)
                viewport_window.set_texture_resolution(*self.sensor_resolution) # 해상도설정
                
                viewport_window.set_window_pos(1000, 400)
                viewport_window.set_window_size(300, 300)

                sd_helper = SyntheticDataHelper()   # 센서 초기화(센서데이터 추출)
                sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                BaseTask.sd_helpers.append(sd_helper)
                BaseTask.viewport_windows.append(viewport_window)
                BaseTask.viewport_handles.append(viewport_handle)
                # viewport_window.set_window_pos(1000, 400)
                # viewport_window.set_window_size(300, 300)

                # # 카메라 생성과 동시에 LeftCamera의 focal length 수정
                # if camera_path == "/World_0/franka/LeftCamera":
                #     left_camera_prim = stage.GetPrimAtPath(camera_path)
                #     if left_camera_prim:
                #         focal_attr = left_camera_prim.GetAttribute("focalLength")
                #         if focal_attr:
                #             original_focal = focal_attr.Get()
                #             new_focal = 20 # original_focal * 2.0  # 현재 값의 80%로 줄임
                #             focal_attr.Set(new_focal)
                #             print(f"Left camera focal length changed from {original_focal} to {new_focal}")

                # sd_helper = SyntheticDataHelper()
                # sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                # BaseTask.sd_helpers.append(sd_helper)
                # BaseTask.viewport_windows.append(viewport_window)
                # BaseTask.viewport_handles.append(viewport_handle)

        else:   # 기존 뷰포트 있으면, 재설정하고 센서 다시 초기화 (1st 빼곤 해당 코드 호출)
            for viewport_handle, camera_path in zip(BaseTask.viewport_handles, self.camera_paths):
                viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window(viewport_handle)
                viewport_window.set_active_camera(camera_path)
                viewport_window.set_texture_resolution(*self.sensor_resolution)
                viewport_window.set_window_pos(1000, 400)
                viewport_window.set_window_size(300, 300)

                sd_helper = SyntheticDataHelper()
                sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                BaseTask.sd_helpers.append(sd_helper)
                BaseTask.viewport_windows.append(viewport_window)
                BaseTask.viewport_handles.append(viewport_handle)
                # viewport_window.set_window_pos(1000, 400)
                # viewport_window.set_window_size(300, 300)

                # # 카메라 생성과 동시에 LeftCamera의 focal length 수정
                # if camera_path == "/World_0/franka/LeftCamera":
                #     left_camera_prim = stage.GetPrimAtPath(camera_path)
                #     if left_camera_prim:
                #         focal_attr = left_camera_prim.GetAttribute("focalLength")
                #         if focal_attr:
                #             original_focal = focal_attr.Get()
                #             new_focal = 20 #original_focal * 2.0  # 현재 값의 80%로 줄임
                #             focal_attr.Set(new_focal)
                #             print(f"Left camera focal length changed from {original_focal} to {new_focal}")

                # BaseTask.viewport_windows.append(viewport_window)
                # sd_helper = SyntheticDataHelper()
                # sd_helper.initialize(sensor_names=self.sensor_types, viewport=viewport_window)
                # BaseTask.sd_helpers.append(sd_helper)

                #============ 디버깅 ============#
                data_name = getattr(self, "current_data_name", "unknown_data")
                prim = stage.GetPrimAtPath(camera_path)
                if not prim:
                    self._log_focal_lengths(data_name=data_name, camera_path=camera_path, error_message="Prim not found")
                else:
                    focal_attr = prim.GetAttribute("focalLength")
                    if not focal_attr:
                        self._log_focal_lengths(data_name=data_name, camera_path=camera_path, error_message="No focalLength attribute")
                    else:
                        focal = focal_attr.Get()
                        if isinstance(focal, (int, float)):
                            if not (11 <= focal <= 13):
                                self._save_viewport_image(sd_helper=sd_helper, viewport_window=viewport_window, camera_path=camera_path)
                                self._log_focal_lengths(data_name=data_name, focal=focal, camera_path=camera_path)
                            # else:
                            #     self._log_focal_lengths(data_name=data_name, focal=focal, camera_path=camera_path)
                        else: 
                            self._log_focal_lengths(data_name=data_name, camera_path=camera_path, error_message=f"Invalid focal value: {focal} (type={type(focal).__name__})")

        self.kit.update()
        self._sensor_initialized = True
    
    # [디버깅] 이미지 저장
    def _save_viewport_image(self, sd_helper, viewport_window, camera_path):
        try:
            gt = sd_helper.get_groundtruth(["rgb"], viewport_window)
            rgb = gt["rgb"]
            rgb = rgb[:, :, :3]  # Remove alpha

            camera_name = camera_path.split("/")[-1].replace("Camera", "")
            save_dir = f"/vagrant/viewport_diff/{camera_name}"
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(rgb.astype(np.uint8)).save(f"{save_dir}/step_{int(time.time())}.png")

        except Exception as e:
            print(f"[ERROR] Failed to save viewport image: {e}")
    
    def _log_focal_lengths(self, data_name="unknown", focal=None, camera_path=None, error_message=None):
        log_path = os.path.abspath("/vagrant/viewport_diff/focal_lengths_log.txt")
        with open(log_path, "a") as log_file:
            if error_message:
                log_file.write(f"{data_name} | {camera_path} → [ERROR] {error_message}\n")
            elif focal is not None:
                log_file.write(f"{data_name} | {camera_path} → focalLength: {focal}\n")
            else:
                log_file.write(f"{data_name} | {camera_path} → [ERROR] Unknown issue\n")

    # [디버깅] focal lengths 로그 저장 함수
    def _log_focal_lengths_all(self, stage):
        log_path = os.path.abspath("/root/arnold/focal_lengths_log.txt")
        with open(log_path, "a") as log_file:
            log_file.write("------ Reset ------\n")
            for camera_path in self.camera_paths:
                prim = stage.GetPrimAtPath(camera_path)
                if prim:
                    focal_attr = prim.GetAttribute("focalLength")
                    if focal_attr:
                        focal = focal_attr.Get()
                        log_file.write(f"{camera_path} → focalLength: {focal}\n")

    def _register_camera_path(self):
        self.camera_paths = []
        
        robot_path = f'/World_{0}/franka'
        camera_paths = [ 
            f'{robot_path}/FrontCamera', 
            f'{robot_path}/BaseCamera',
            f'{robot_path}/LeftCamera',
            f'{robot_path}/panda_hand/GripperCameraBottom',
            f'{robot_path}/panda_hand/GripperCamera'
        ]

        for camera_path in camera_paths:
            self.camera_paths.append(camera_path)

    def _wait_for_loading(self):
        sim = SimulationContext.instance()
        sim.render()
        while is_stage_loading():
            sim.render()

    def register_recorder(self):
        index = 0
        objects_paths = [prim.GetPath().pathString for prim in self.objects_list]
        self.recorder = DataRecorder(self.robot.prim_path, objects_paths, self.robot, self.scene_parameters[index].task_type)

    def try_record(self, actions):
        if self.recorder is not None and self.recorder.record:
            # dof_states = self.dc.get_articulation_dof_states(self.articulation, _dynamic_control.STATE_ALL)
            dof_states = {
                'pos': self.robot.get_joint_positions(),
                'vel': self.robot.get_joint_velocities(),
                # 'effort': self.robot.get_joint_efforts(),   # get_measured_joint_efforts() for Isaac Sim 2023
                # getting effort has conflict with use_gpu_dynamics
            }
            self.recorder.record_data(
                robot_states=dof_states,
                actions=actions,
                time_step=self.time_step,
            )
