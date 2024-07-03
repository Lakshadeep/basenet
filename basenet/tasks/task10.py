# Isaac imports

from omni.isaac.kit import SimulationApp


# This enables a livestream server to connect to when running headless
# CONFIG = {
#     "width": 1280,
#     "height": 720,
#     "window_width": 1920,
#     "window_height": 1080,
#     "headless": False,
#     "renderer": "RayTracedLighting",
#     # "display_options": 3286,  # Set display options to show default grid,
# }

CONFIG = {
    "width": 640,
    "height": 480,
    "window_width": 640,
    "window_height": 480,
    "headless": True,
    "renderer": "RayTracedLighting",
    # "display_options": 3286,  # Set display options to show default grid,
}

# Start the omniverse application
simulation_app = SimulationApp(launch_config=CONFIG)

# Default Livestream settings
simulation_app.set_setting("/app/window/drawMouse", True)
simulation_app.set_setting("/app/livestream/proto", "ws")
simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 10)
simulation_app.set_setting("/ngx/enabled", False)


import omni.kit
# from utils.synthetic_data import SyntheticDataHelper
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.dynamic_control import _dynamic_control
import omni.graph.core as og
from omni.isaac.core_nodes.scripts.utils import set_target_prims    
from omni.isaac.core.utils.extensions import enable_extension
from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics

from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula import LulaTaskSpaceTrajectoryGenerator
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper


# general python libraries
import gym
from gym import spaces
import torch
import math
import numpy as np
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
import datetime
from datetime import time
import cv2
import time
# import random
import networkx as nx
from karateclub.graph_embedding import Graph2Vec
from gensim.models.doc2vec import Doc2Vec
from shapely.geometry import Polygon



# Mushroom rl imports
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.angles import normalize_angle
from mushroom_rl.utils.viewer import Viewer
from mushroom_rl.utils.spaces import *


# enable websocket extension
# enable_extension("omni.services.streamclient.websocket")


class Task(Environment):
    """
    Simple room environment for ER mobile manipulator and onrobotic 2f gripper
    """

    ur5e_default_joint_angles = [-180,-80,105,-155,-95,-180]

    def __init__(
        self ,
        cfg
    ) -> None:

        """
        Constructor.
        Args:
            horizon (int, 5): horizon of the problem;
            gamma (float, .95): discount factor.
        """
        self.cfg = cfg

        # MDP properties
        gamma = cfg.task.mdp.gamma                                
        horizon = cfg.task.mdp.horizon


        if self.cfg.task.subtask == 'single_obj':
            observation_space = spaces.Box(low=np.array([self.cfg.task.mdp.table_dimensions.x_min, 
                                            self.cfg.task.mdp.table_dimensions.y_min,-np.pi,
                                            -2, -2, -np.pi]), 
                                high=np.array([self.cfg.task.mdp.table_dimensions.x_max, 
                                            self.cfg.task.mdp.table_dimensions.y_max,np.pi,
                                            2, 2, np.pi]))
        elif self.cfg.task.subtask == 'sequence' or self.cfg.task.subtask == 'sequence_nc':
            # robot, objects
            observation_space = spaces.Box(low=np.array([-2, -2, -np.pi, 0,
                                                         self.cfg.task.mdp.table_dimensions.x_min, self.cfg.task.mdp.table_dimensions.y_min, -np.pi, 0,
                                                         self.cfg.task.mdp.table_dimensions.x_min, self.cfg.task.mdp.table_dimensions.y_min, -np.pi, 0,
                                                         self.cfg.task.mdp.table_dimensions.x_min, self.cfg.task.mdp.table_dimensions.y_min, -np.pi, 0,
                                                         self.cfg.task.mdp.table_dimensions.x_min, self.cfg.task.mdp.table_dimensions.y_min, -np.pi, 0,
                                                         self.cfg.task.mdp.table_dimensions.x_min, self.cfg.task.mdp.table_dimensions.y_min, -np.pi, 0]), 
                                            high=np.array([2, 2, np.pi, 0,
                                                        self.cfg.task.mdp.table_dimensions.x_max, self.cfg.task.mdp.table_dimensions.y_max, np.pi, 1,
                                                        self.cfg.task.mdp.table_dimensions.x_max, self.cfg.task.mdp.table_dimensions.y_max, np.pi, 1,
                                                        self.cfg.task.mdp.table_dimensions.x_max, self.cfg.task.mdp.table_dimensions.y_max, np.pi, 1,
                                                        self.cfg.task.mdp.table_dimensions.x_max, self.cfg.task.mdp.table_dimensions.y_max, np.pi, 1,
                                                        self.cfg.task.mdp.table_dimensions.x_max, self.cfg.task.mdp.table_dimensions.y_max, np.pi, 1]))

        action_space = None

        if self.cfg.task.subtask == 'single_obj':
            action_space = spaces.Box(low=np.array([-0.85, -0.85, -np.pi]),
                                    high=np.array([0.85, 0.85, np.pi]))
            # action_space = spaces.Box(low=np.array([-0.9, -1.5, -np.pi]),
            #                         high=np.array([0.9, 1.5, np.pi]))
        elif self.cfg.task.subtask == 'sequence':
            action_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                    high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
            # action_space = spaces.Box(low=np.array([0]),
                                    # high=np.array([1]))
        elif self.cfg.task.subtask == 'sequence_nc':
            action_space = spaces.Box(low=np.array([0, 0]),
                                    high=np.array([1, 1]))
             
        
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        self.initialize_mdp(mdp_info)
        self.initialize_parameters()

        # initializes the episode scene
        self._set_up_scene()


    def shutdown(self):
        simulation_app.close()


    def _bound(x, min_value, max_value):
        print("Using custom bound")
        """
        Method used to bound state and action variables.

        Args:
            x: the variable to bound;
            min_value: the minimum value;
            max_value: the maximum value;

        Returns:
            The bounded variable.

        """
        return np.maximum(min_value, np.minimum(x, max_value))


    def initialize_graph(self):
        self.graph_embeddings = Graph2Vec(attributed=False, dimensions=self.cfg.graph_embeddings.train.dimensions)
        self.graph_embeddings.model = Doc2Vec.load(self.cfg.graph_embeddings.test.load_model) 


    def initialize_mdp(self, mdp_info):
        super().__init__(mdp_info)


    def initialize_parameters(self):
        # task-specific parameters
        self._robot_base_pose = [0.0, 0.0, 2.0] # x, y, theta

        # Initialize Isaac API for getting synthetic data
        # self.synthetic_data = SyntheticDataHelper()
        # self.viewport_interface = omni.kit.viewport_legacy.get_viewport_interface()

        # some private variables
        self._state = np.zeros((24,))
        self._random = True
        self._n_steps = 0
        self.obj_list = []

        # UR5e default joint angles in degrees
        self.ur5e_default_joint_angles = [-180,-80,105,-155,-95,-180]

        # UR5e joint limits in degrees
        # lower bound
        self.ur5e_joint_lower_limits = [-2*np.pi, -np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi]

        # upper boound
        self.ur5e_joint_upper_limits = [2*np.pi, 0, np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
        


    def hard_reset(self):
        # start simulation
        self.world.reset()

        # Initialization should happen only after simulation starts
        self.robot.initialize()


    def reset(self, state=None, robot_r_max=3.0, robot_r_min=1.8, safety_dist=0.1):
        # print("New episode")
        table_length = 2*self.cfg.task.mdp.table_dimensions.y_max
        table_width = 2*self.cfg.task.mdp.table_dimensions.x_max

        self.epsiode_length = 0
        self._n_steps = 0
        self.reset_arm()

        # without this random doesnt work TODO find out why it worked in previous problems?
        np.random.seed()


        self.obj_list = np.array([True, True, True, True, True, True, True, True, True, True])
        if self.cfg.task.subtask == 'single_obj':
            self.obj_list = np.random.choice([False, True], len(self.obj_prims))
            

        # we ensure that there is atleast one object
        while(1):
            if np.any(self.obj_list):
                break
            else:
                self.obj_list = np.random.choice([True, False], len(self.obj_prims))


        if self.cfg.task.subtask == 'single_obj':
            # TODO select appropriate object
            self.obj_list[0] = [True]

        # if self.cfg.task.subtask == 'single_sequence':
        #     self.obj_list = np.array([True, True, True, True, True])
        #     obj_pos = np.array([[-0.3, 0.8, np.pi], [0.2, 0.6, np.pi], [0.3, -0.8, -np.pi], [-0.3, 0.4, -np.pi/2], [0.1, -0.2, -np.pi/8]])
        
        # obj_pos = np.array([[-0.3, 0.8, np.pi], [0.2, 0.6, np.pi], [0.3, -0.8, -np.pi], [-0.3, 0.4, -np.pi/2], [0.1, -0.2, -np.pi/8]])

        self.epsiode_length = self.obj_list.sum()

        if state is None:            
            for i, status in enumerate(self.obj_list):
                if status:
                    self.obj_prims[i].set_visibility(True)
                    obj_y = np.random.uniform(-table_length/2.0 + safety_dist, table_length/2.0 - safety_dist)
                    obj_x = np.random.uniform(-table_width/2.0 + safety_dist, table_width/2.0 - safety_dist)

                    obj_theta = np.random.uniform(-np.pi, np.pi)

                    # if self.cfg.task.subtask == 'single_sequence':
                    #     obj_y = obj_pos[i,1]
                    #     obj_x = obj_pos[i,0]
                    #     obj_theta = obj_pos[i,2]

                    # obj_y = obj_pos[i,1]
                    # obj_x = obj_pos[i,0]
                    # obj_theta = obj_pos[i,2]
                    
                    self._move_obj_to_pose(obj_x, obj_y, obj_theta, self.obj_prims[i])
                else:
                    self.obj_prims[i].set_visibility(False)
            
            while True:
                if(not self.check_collision()):
                    break         
                    
                robot_x = np.random.uniform(-1.4, 1.4)
                robot_y = np.random.uniform(-1.9, 1.9)

                # fixed orientations
                robot_thetas = np.array([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
                robot_theta = np.random.choice(robot_thetas,1)[0]
                r_theta = robot_theta + np.random.uniform(-np.pi/12, np.pi/12)

                self._move_robot_base_to_pose(robot_x, robot_y, r_theta)     
                  
        else:
            # TODO
            pass

        self.no_of_objs_to_grasp = self.obj_list.sum()
        # print("Number of objects to grasp:", self.no_of_objs_to_grasp)
        
        self.world.step(render=self.cfg.task.mdp.render)

        self._state = self._get_state()
        # print("State:", self._state)
        return self._state

    
    def transform_all_poses_to_table_frame(self, table_pose, robot_base_pose, object_poses):
        r_pose_t = np.zeros((3,))
        obj_poses_t = np.zeros((len(object_poses), 3))

        wTt = self._isaac_pose_to_transformation_matrix(table_pose)
        wTr = self._isaac_pose_to_transformation_matrix(robot_base_pose)

        tTr = np.matmul(np.linalg.inv(wTt), wTr)

        r_rot_se2 = Rotation.from_matrix(tTr[:3,:3]).as_euler('xyz')
        r_quat = Rotation.from_matrix(tTr[:3,:3]).as_quat()
        r_pose_t = [tTr[0,3], tTr[1,3], r_rot_se2[2]]
        # r_pose_t = [tTr[0,3], tTr[1,3], tTr[2,3], r_quat[0], r_quat[1], r_quat[2], r_quat[3]]


        for i, object_pose in enumerate(object_poses):
            wTo = self._isaac_pose_to_transformation_matrix(object_pose)
            tTo = np.matmul(np.linalg.inv(wTt), wTo)

            o_rot_se2 = Rotation.from_matrix(tTo[:3,:3]).as_euler('xyz')
            o_quat = Rotation.from_matrix(tTo[:3,:3]).as_quat()
            obj_poses_t[i,:] = [tTo[0,3], tTo[1,3], o_rot_se2[2]]
            # obj_poses_t[i,:] = [tTo[0,3], tTo[1,3], tTo[2,3], o_quat[0], o_quat[1], o_quat[2], o_quat[3]]
        return r_pose_t, obj_poses_t

    
    def get_2D_robot_pose_in_table_frame(self):
        table_pose = self.table_prim.get_world_pose()
        robot_base_pose = self.robot_base_prim.get_world_pose()
        r_pose_t = np.zeros((3,))

        wTt = self._isaac_pose_to_transformation_matrix(table_pose)
        wTr = self._isaac_pose_to_transformation_matrix(robot_base_pose)

        tTr = np.matmul(np.linalg.inv(wTt), wTr)

        r_rot_se2 = Rotation.from_matrix(tTr[:3,:3]).as_euler('xyz')
        r_quat = Rotation.from_matrix(tTr[:3,:3]).as_quat()
        r_pose_t = [tTr[0,3], tTr[1,3], r_rot_se2[2]]
        return r_pose_t


    def check_collision(self):
        robot_pose = self.robot_base_prim.get_world_pose()
        table_pose = self.table_prim.get_world_pose()

        robot_rot_rpy = Rotation.from_quat([robot_pose[1][1], robot_pose[1][2], robot_pose[1][3], robot_pose[1][0]]).as_euler('xyz')
        table_rot_rpy = Rotation.from_quat([table_pose[1][1], table_pose[1][2], table_pose[1][3], table_pose[1][0]]).as_euler('xyz')


        tl = self.cfg.task.mdp.table_dimensions.y_max + self.cfg.task.mdp.safe_dist_from_table
        tw = self.cfg.task.mdp.table_dimensions.x_max + self.cfg.task.mdp.safe_dist_from_table
        t = pow(pow(tl,2) + pow(tw,2), 0.5)
        ang = math.atan(tl/tw)
        table_c1 = [table_pose[0][0] + (t*np.cos(self._wrap_angle(table_rot_rpy[2] + ang))), table_pose[0][1] + (t*np.sin(self._wrap_angle(table_rot_rpy[2] + ang))) ]
        table_c2 = [table_pose[0][0] + (t*np.cos(self._wrap_angle(table_rot_rpy[2] - ang))), table_pose[0][1] + (t*np.sin(self._wrap_angle(table_rot_rpy[2] - ang))) ]
        table_c3 = [table_pose[0][0] + (t*np.cos(self._wrap_angle(table_rot_rpy[2] + ang - np.pi))), table_pose[0][1] + (t*np.sin(self._wrap_angle(table_rot_rpy[2] + ang - np.pi)))]
        table_c4 = [table_pose[0][0] + (t*np.cos(self._wrap_angle(table_rot_rpy[2] - ang + np.pi))), table_pose[0][1] + (t*np.sin(self._wrap_angle(table_rot_rpy[2] - ang + np.pi)))]
        
        table = [table_c1, table_c2, table_c3, table_c4, table_c1]

        rl = 0.8/2
        rw = 0.5/2
        r = pow(pow(rl,2) + pow(rw,2), 0.5)
        ang = math.atan(rl/rw) 
        r_off = np.pi/2

        robot_c1 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] + ang + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] + ang + r_off))) ]
        robot_c2 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] - ang + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] - ang + r_off))) ]
        robot_c3 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] + ang - np.pi + r_off)))]
        robot_c4 = [robot_pose[0][0] + (r*np.cos(self._wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off))), robot_pose[0][1] + (r*np.sin(self._wrap_angle(robot_rot_rpy[2] - ang + np.pi + r_off)))]
        
        robot = [robot_c1, robot_c2, robot_c3, robot_c4, robot_c1]

        table_polygon = Polygon(table)
        robot_polygon = Polygon(robot)
        collision_status = table_polygon.intersects(robot_polygon) or table_polygon.contains(robot_polygon)
        # print("Collision status:", collision_status)
        

        return collision_status



    # TODO Now state will also require knowledge of previous (successful) actions

    def _get_state(self):
        if self.cfg.task.subtask == 'single_obj':
            table_pose = self.table_prim.get_world_pose()
            robot_base_pose = self.robot_base_prim.get_world_pose()
            cracker_box_pose = self.cracker_box_prim.get_world_pose()
            robot_pose_t, obj_poses_t = self.transform_all_poses_to_table_frame(table_pose, robot_base_pose, [cracker_box_pose])
            state = np.hstack((obj_poses_t[0,0:3], robot_pose_t[0:3]))
            return state
        elif self.cfg.task.subtask == 'sequence' or self.cfg.task.subtask == 'single_sequence':
            table_pose = self.table_prim.get_world_pose()
            robot_base_pose = self.robot_base_prim.get_world_pose()
            cracker_box_pose = self.cracker_box_prim.get_world_pose()
            mustard_bottle_pose = self.mustard_bottle_prim.get_world_pose()
            bleach_cleanser_pose = self.bleach_cleanser_prim.get_world_pose()
            mug_pose = self.mug_prim.get_world_pose()
            bowl_pose = self.bowl_prim.get_world_pose()
            cracker_box_pose1 = self.cracker_box_prim1.get_world_pose()
            mustard_bottle_pose1 = self.mustard_bottle_prim1.get_world_pose()
            bleach_cleanser_pose1 = self.bleach_cleanser_prim1.get_world_pose()
            mug_pose1 = self.mug_prim1.get_world_pose()
            bowl_pose1 = self.bowl_prim1.get_world_pose()
            robot_pose_t, obj_poses_t = self.transform_all_poses_to_table_frame(table_pose, robot_base_pose, 
                                        [cracker_box_pose, bleach_cleanser_pose, mustard_bottle_pose, mug_pose, bowl_pose,
                                        cracker_box_pose1, bleach_cleanser_pose1, mustard_bottle_pose1, mug_pose1, bowl_pose1])

            state = np.zeros((44,))

            # robot state
            # print("Robot pose t:", robot_pose_t)
            state[0:3] = robot_pose_t
            state[3] = 0

            n_start = 4
            for idx, obj_status in enumerate(self.obj_list):
                # if obj_status:
                obj_base_pose = obj_poses_t[idx]
                state[n_start:n_start+3] = obj_base_pose
                state[n_start + 3] = obj_status
                n_start = n_start + 4
                

            # print("State:", state)
            return state.astype(np.float32)
        

        print("Invalid subtask")
        return None



    def get_arm_joint_angles(self):
        joint_positions = self.ur5e_sm.get_joint_positions()
        # print("Arm joints:", np.rad2deg(joint_positions[0:6]))
        return joint_positions[0:6]
    
    
    def step(self, action):
        # TODO

        self.world.step(render=self.cfg.task.mdp.render)


    def stop(self):
        # simulation_app.close()
        pass


    def _move_robot_base_to_pose(self, x, y, theta):
        orientation_rpy = [0, 0, theta]
        base_q = Rotation.from_euler('xyz', orientation_rpy, degrees=False).as_quat()         

        current_robot_base_pose = self.robot_base_prim.get_world_pose()
        current_robot_arm_base_pose = self.ur5e_prim.get_world_pose()

        aTb = self._get_transformation_matrix(current_robot_arm_base_pose, current_robot_base_pose)
        wTb_1 = self._pose_to_transformation_matrix([x, y, self.robot_base_initial_pose[0][2], base_q[0], base_q[1], base_q[2], base_q[3]])

        aTw = np.matmul(aTb, np.linalg.inv(wTb_1))
        wTa = np.linalg.inv(aTw)
        arm_q = Rotation.from_matrix(wTa[:3,:3]).as_quat()

        self.robot_base_prim.set_world_pose(position = np.array([x, y, self.robot_base_initial_pose[0][2]]),
                                      orientation = np.array([base_q[3], base_q[0], base_q[1], base_q[2]]))
        
        self.ur5e_prim.set_world_pose(position = np.array([wTa[0][3], wTa[1][3], self.ur5e_initial_pose[0][2]]),
                                      orientation = np.array([arm_q[3], arm_q[0], arm_q[1], arm_q[2]]))
        self.world.step(render=self.cfg.task.mdp.render)
        # self.world.step(render=self.cfg.task.mdp.render)
        # self.world.step(render=self.cfg.task.mdp.render)
        # self.world.step(render=self.cfg.task.mdp.render)
        # self.world.step(render=self.cfg.task.mdp.render)

    def _move_robot(self, dist, theta, j1 = ur5e_default_joint_angles[0], 
                                       j2 = ur5e_default_joint_angles[1],
                                       j3 = ur5e_default_joint_angles[2],
                                       j4 = ur5e_default_joint_angles[3],
                                       j5 = ur5e_default_joint_angles[4],
                                       j6 = ur5e_default_joint_angles[5]
                    ):
        current_robot_base_pose = self.robot_base_prim.get_world_pose()
        current_robot_arm_base_pose = self.ur5e_prim.get_world_pose()
        
        # NOTE: NVIDIA quaternion format [w,x,y,z]
        quat_base = current_robot_base_pose[1]
        curr_base_theta = Rotation.from_quat([quat_base[1], quat_base[2], quat_base[3], 
            quat_base[0]]).as_euler('xyz', degrees=False)[2]

        # NOTE: first forward motion and then turn
        new_base_x = current_robot_base_pose[0][0] + (dist * np.cos(curr_base_theta))
        new_base_y = current_robot_base_pose[0][1] + (dist * np.sin(curr_base_theta))
        new_base_theta = self._wrap_angle(curr_base_theta + theta)

        self._move_robot_base_to_pose(new_base_x, new_base_y, new_base_theta)

        # angles should be in degrees
        self.move_arm(j1, j2, j3, j4, j5, j6)

  
    
    def _move_obj_to_pose(self, x, y, theta, obj_prim):
        table_pose = self.table_prim.get_world_pose()
        obj_pose = obj_prim.get_world_pose()   

        obj_rot_rpy = Rotation.from_quat([obj_pose[1][1], obj_pose[1][2], obj_pose[1][3], obj_pose[1][0]]).as_euler('xyz')
        obj_rot_rpy[2] = theta
        quat = Rotation.from_euler('xyz', obj_rot_rpy, degrees=False).as_quat()  


        obj_prim.set_world_pose(position = np.array([table_pose[0][0] + x, table_pose[0][1] + y, 
                                obj_pose[0][2]]), orientation = np.array([quat[3], quat[0], quat[1], quat[2]]))
        self.world.step(render=self.cfg.task.mdp.render)



    def _wrap_angle(self, angle):
        angle = math.fmod(angle, 2 * np.pi)
        if (angle >= np.pi):
            angle -= 2 * np.pi
        elif (angle <= -np.pi):
            angle += 2 * np.pi
        return angle


    def _isaac_pose_to_transformation_matrix(self, pose):
        r = Rotation.from_quat([pose[1][1], pose[1][2], pose[1][3], pose[1][0]])
        T = np.identity(4)
        T[:3,:3] = r.as_matrix()
        T[0,3] = pose[0][0] 
        T[1,3] = pose[0][1] 
        T[2,3] = pose[0][2]
        return T

    def _pose_to_transformation_matrix(self, pose):
        r = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])
        T = np.identity(4)
        T[:3,:3] = r.as_matrix()
        T[0,3] = pose[0] 
        T[1,3] = pose[1] 
        T[2,3] = pose[2]
        return T

    def _pose_to_isaac_pose(self, pose):
        return (np.array([pose[0], pose[1], pose[2]]), np.array([pose[6], pose[3], pose[4], pose[5]]))

    def _get_transformation_matrix(self, pose_s, pose_d):
        wTs = self._isaac_pose_to_transformation_matrix(pose_s)
        wTd = self._isaac_pose_to_transformation_matrix(pose_d)
        sTw = np.linalg.inv(wTs)
        sTd = np.matmul(sTw, wTd)
        return sTd

    def _deg_to_rad(self, angle):
        return angle * np.pi/180.0


    def _get_reward(self):
        reward = 0
        goal_status = False
        return reward, goal_status


    def reset_arm(self):
        self.ur5e_sm.set_joint_positions(positions=np.deg2rad(self.ur5e_default_joint_angles), joint_indices=np.array([0,1,2,3,4,5]))


    def _set_up_scene(self) -> None:
        assets_root_path = get_assets_root_path()

        # TODO: make this path relative
        scene_path = self.cfg.task.env_file
        open_stage(usd_path=scene_path)

        self._usd_context = omni.usd.get_context()
        self._stage = self._usd_context.get_stage()

        self.world = World()

        self.world.scene.add(Robot(prim_path="/World/ur5e", name="ur5e"))

        self.robot = Articulation("/World/ur5e")

        # initialize isaac prims for actors in the scene
        
        self.ur5e_prim = XFormPrim(prim_path="/World/ur5e", name="ur5e")
        self.ur5e_initial_pose = self.ur5e_prim.get_world_pose()
        self.ur5e_curr_pose = self.ur5e_prim.get_world_pose()

        gripper = ParallelGripper(
            #We chose the following values while inspecting the articulation
            end_effector_prim_path="/World/ur5e/onrobot_rg6_base_link",
            joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([0.628, -0.628]),
            action_deltas=np.array([-0.628, 0.628]),
        )

        self.ur5e_sm = self.world.scene.add(SingleManipulator(prim_path="/World/ur5e", name="ur5e_sm", \
            end_effector_prim_name="onrobot_rg6_base_link", gripper=gripper))


        self.robot_base_prim = XFormPrim(prim_path="/World/Base", name="Base")
        self.robot_base_initial_pose = self.robot_base_prim.get_world_pose()
        self.robot_base_pose = self.robot_base_prim.get_world_pose()

        self.table_prim = XFormPrim(prim_path="/World/Table", name="Table")
        self.table_initial_pose = self.table_prim.get_world_pose()
        self.table_pose = self.table_prim.get_world_pose()

        self.robot_base_top_prim = XFormPrim(prim_path="/World/BaseTop", name="BaseTop")
        self.robot_base_top_initial_pose = self.robot_base_top_prim.get_world_pose()
        self.robot_base_top_pose = self.robot_base_top_prim.get_world_pose()

        # obj 0
        self.cracker_box_prim = XFormPrim(prim_path="/World/_03_cracker_box", name="_03_cracker_box")
        self.cracker_box_initial_pose = self.cracker_box_prim.get_world_pose()
        self.cracker_box_pose = self.cracker_box_prim.get_world_pose()

        # obj 1
        self.bleach_cleanser_prim = XFormPrim(prim_path="/World/_21_bleach_cleanser", name="_21_bleach_cleanser")
        self.bleach_cleanser_initial_pose = self.bleach_cleanser_prim.get_world_pose()
        self.bleach_cleanser_pose = self.bleach_cleanser_prim.get_world_pose()

        # obj 2
        self.mustard_bottle_prim = XFormPrim(prim_path="/World/_06_mustard_bottle", name="_06_mustard_bottle")
        self.mustard_bottle_initial_pose = self.mustard_bottle_prim.get_world_pose()
        self.mustard_bottle_pose = self.mustard_bottle_prim.get_world_pose()

        # obj 3
        self.mug_prim = XFormPrim(prim_path="/World/_25_mug", name="_25_mug")
        self.mug_initial_pose = self.mug_prim.get_world_pose()
        self.mug_pose = self.mug_prim.get_world_pose()

        # obj 4
        self.bowl_prim = XFormPrim(prim_path="/World/_24_bowl", name="_24_bowl")
        self.bowl_initial_pose = self.bowl_prim.get_world_pose()
        self.bowl_pose = self.bowl_prim.get_world_pose()

        # obj 5
        self.cracker_box_prim1 = XFormPrim(prim_path="/World/_03_cracker_box_01", name="_03_cracker_box_01")
        self.cracker_box_initial_pose1 = self.cracker_box_prim1.get_world_pose()
        self.cracker_box_pose1 = self.cracker_box_prim.get_world_pose()

        # obj 6
        self.bleach_cleanser_prim1 = XFormPrim(prim_path="/World/_21_bleach_cleanser_01", name="_21_bleach_cleanser_01")
        self.bleach_cleanser_initial_pose1 = self.bleach_cleanser_prim1.get_world_pose()
        self.bleach_cleanser_pose1 = self.bleach_cleanser_prim1.get_world_pose()

        # obj 7
        self.mustard_bottle_prim1 = XFormPrim(prim_path="/World/_06_mustard_bottle_01", name="_06_mustard_bottle_01")
        self.mustard_bottle_initial_pose1 = self.mustard_bottle_prim1.get_world_pose()
        self.mustard_bottle_pose1 = self.mustard_bottle_prim1.get_world_pose()

        # obj 8
        self.mug_prim1 = XFormPrim(prim_path="/World/_25_mug_01", name="_25_mug_01")
        self.mug_initial_pose1 = self.mug_prim1.get_world_pose()
        self.mug_pose1 = self.mug_prim1.get_world_pose()

        # obj 9
        self.bowl_prim1 = XFormPrim(prim_path="/World/_24_bowl_01", name="_24_bowl_01")
        self.bowl_initial_pose1 = self.bowl_prim1.get_world_pose()
        self.bowl_pose1 = self.bowl_prim1.get_world_pose()

        self.obj_prims = [self.cracker_box_prim, self.bleach_cleanser_prim, self.mustard_bottle_prim, self.mug_prim, self.bowl_prim,
                            self.cracker_box_prim1, self.bleach_cleanser_prim1, self.mustard_bottle_prim1, self.mug_prim1, self.bowl_prim1]

        self.camera_prim = XFormPrim(prim_path="/World/ur5e/tool0/Camera", name="Camera")
        self.camera_initial_pose = self.camera_prim.get_world_pose()
        self.camera_pose = self.camera_prim.get_world_pose()

        self.tool0_prim = XFormPrim(prim_path="/World/ur5e/tool0", name="tool0")
        self.tool0_initial_pose = self.tool0_prim.get_world_pose()
        self.tool0_pose = self.tool0_prim.get_world_pose()
        
    
        # start simulation
        self.world.reset()

        # Initialization should happen only after simulation starts
        self.robot.initialize()

        # Lula Kinematics solver
        self.lula_kinematics_solver = LulaKinematicsSolver(
            robot_description_path = '/home/sdur/Planning/Codes/basenet/ur5e_assets/robot_descriptor_basic.yaml',
            urdf_path = '/home/sdur/Planning/Codes/basenet/ur5e_assets/ur5e_gripper.urdf'
        )

        self.articulation_kinematics_solver = ArticulationKinematicsSolver(self.robot, self.lula_kinematics_solver, "tool0")

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self.task_space_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path = '/home/sdur/Planning/Codes/basenet/ur5e_assets/robot_descriptor_basic.yaml',
            urdf_path = '/home/sdur/Planning/Codes/basenet/ur5e_assets/ur5e_gripper.urdf'
        )

        lower_limits = np.array([-2*np.pi,-np.pi,-2*np.pi,-2*np.pi,-2*np.pi,-2*np.pi])
        upper_limits = np.array([2*np.pi,0,2*np.pi,2*np.pi,2*np.pi,2*np.pi])
        # self.task_space_trajectory_generator.set_c_space_position_limits(lower_limits, upper_limits)

        print("Simulation start status: ", self.world.is_playing())


        if self.world.is_playing():
            # self.ur5e_sm.set_joint_positions(positions=np.deg2rad(self.ur5e_default_joint_angles), joint_indices=np.array([0,1,2,3,4,5]))
            pass

        self.reset()
        
        # status = True       
        # while simulation_app.is_running():
        #     self.world.step(render=self.cfg.task.mdp.render)

        #     if self.world.current_time_step_index == 0:
        #         self.world.reset()
        #     else:                
        #         self.reset()
        #         time.sleep(1)

        # simulation_app.close()
