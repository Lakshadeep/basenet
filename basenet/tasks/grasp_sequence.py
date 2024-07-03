# from basenet.tasks.task10 import Task
from basenet.tasks.task import Task

import numpy as np
import math
from scipy.spatial.transform import Rotation
import time
from shapely.geometry import Polygon


from mushroom_rl.core import MDPInfo
from mushroom_rl.utils import spaces

from pxr import UsdLux, UsdGeom, Sdf, Gf, UsdPhysics
from omni.isaac.core.prims import XFormPrim

from basenet.utils.navigation_cost import Navigation


class GraspSequence(Task):
    """
    Determining grasp sequence
    """

    def __init__(
        self, cfg
    ) -> None:
        if cfg.task.test.save_data:
            self.obj_status_init = np.array([])   
            self.obj_status_final = np.array([])     

        super().__init__(cfg)

        self.navigation = Navigation(cfg)

        self.robot_init_pose_t = None
        self.predicted_base_pose_t = [0,0,0]

        self.navigation_costs = []
        self.grasping_costs = []


    
    def load_prior_agent(self, prior_agent):
        self._prior_policy = prior_agent.policy

    
    def get_base_pose(self, robot_state, obj_state):
        state = np.hstack((obj_state, robot_state))
        x, y, theta = self._prior_policy.draw_action(state)
        
        # from relative to world
        T_oa = np.eye(4)
        T_oa[:3,:3] = Rotation.from_euler('z', theta).as_matrix()
        T_oa[0,3] = x
        T_oa[1,3] = y

        T_to = np.eye(4)
        T_to[:3,:3] = Rotation.from_euler('z', obj_state[2]).as_matrix()
        T_to[0,3] = obj_state[0]
        T_to[1,3] = obj_state[1]

        T_ta = np.matmul(T_to, T_oa)
        x = T_ta[0,3]
        y = T_ta[1,3]
        theta = Rotation.from_matrix(T_ta[:3,:3]).as_euler('xyz')[2]

        return np.array([x, y, theta])


    def reset(self, state=None):
        self.predicted_base_pose_t = [0,0,0]
        self.robot_init_pose_t = None

        state = super().reset()

        self.objs_to_grasp = self.obj_list
        # print("Objects to grasp:", self.objs_to_grasp)
        self._n_steps = 0
        self.objs_to_grasp_action = [0,0,0,0,0]
        self.no_of_objs_to_grasp = 0

        if self.cfg.task.test.save_data:
            self.obj_status_init = np.append(self.obj_status_init, self.obj_list)

        self.robot_init_pose_t = self.get_2D_robot_pose_in_table_frame()    

        # print("Objs to grasp:", self.objs_to_grasp)
        return state

    
    def table_to_world_frame(self, x, y, theta):
        table_pose = self.table_prim.get_world_pose()

        robot_rot_in_quat = Rotation.from_euler('z', theta).as_quat()

        tTr = self._pose_to_transformation_matrix(np.hstack(([x,y,table_pose[0][2]], robot_rot_in_quat)))
        wTt = self._isaac_pose_to_transformation_matrix(table_pose)

        r_pose_t = np.zeros((3,))

        wTr = np.matmul(np.linalg.inv(wTt), tTr)

        r_rot_se2 = Rotation.from_matrix(wTr[:3,:3]).as_euler('xyz')
        r_pose_t = [wTr[0,3], wTr[1,3], r_rot_se2[2]]
        # print("Pose in table frame:", r_pose_t)
        return r_pose_t

    
    def get_label(self):
        navigation_costs = []
        n = 0
        for idx, obj in enumerate(self.obj_list):
            if obj:
                obj_start_idx = (idx*4) + 4
                obj_end_idx = obj_start_idx + 3
                self.predicted_base_pose_t = self.get_base_pose(self._state[0:3], self._state[obj_start_idx:obj_end_idx])
                
                start = self.get_2D_robot_pose_in_table_frame()
                end = self.predicted_base_pose_t
                nav_cost = self.navigation.compute_path(start, end)
                navigation_costs.append(nav_cost)
                n = n + 1
        # print("Navigation costs:", navigation_costs)
        navigation_costs = np.array(navigation_costs)
        return np.argmin(navigation_costs)


    def get_baseline_reward(self):
        navigation_costs = []
        n = 0
        for idx, obj in enumerate(self.obj_list):
            if obj:
                obj_start_idx = (idx*4) + 4
                obj_end_idx = obj_start_idx + 3
                self.predicted_base_pose_t = self.get_base_pose(self._state[0:3], self._state[obj_start_idx:obj_end_idx])
                
                start = self.get_2D_robot_pose_in_table_frame() # this can be outside for loop right?
                end = self.predicted_base_pose_t
                nav_cost = self.navigation.compute_path(start, end)
                navigation_costs.append(nav_cost)
                n = n + 1
        # print("Navigation costs:", navigation_costs)
        navigation_costs = np.array(navigation_costs)

        reward = 0
        if len(navigation_costs) > 0:
            action_idx = np.argmin(navigation_costs)
            # reward = 100000/(1+navigation_costs[action_idx])
            reward = - (1000 * navigation_costs[action_idx])
        return reward

    def get_updated_baseline_reward(self, action_idx):
        navigation_cost = 0
        n = 0
        for idx, obj in enumerate(self.obj_list):
            if obj:
                if idx == action_idx:
                    obj_start_idx = (idx*4) + 4
                    obj_end_idx = obj_start_idx + 3
                    self.predicted_base_pose_t = self.get_base_pose(self._state[0:3], self._state[obj_start_idx:obj_end_idx])
                    
                    start = self.get_2D_robot_pose_in_table_frame()
                    end = self.predicted_base_pose_t
                    nav_cost = self.navigation.compute_path(start, end)
                    navigation_cost = nav_cost
                    n = n + 1
        
        reward = - (1000 * navigation_cost)
        return reward


    def step(self, action):
        self.robot_init_pose_t = self.get_2D_robot_pose_in_table_frame()    
        # print("Step no:", self._n_steps)
        # print("Action in step:", action)

        self.objs_to_grasp_action = action[0:len(self.obj_list)]
        # print("Objects to grasp:", self.objs_to_grasp_action)

        n = 0
        self.obj_exist = 0
        for idx, obj in enumerate(self.obj_list):
            if obj:
                if self.objs_to_grasp_action[n] == 1:
                    obj_start_idx = (idx*4) + 4
                    obj_end_idx = obj_start_idx + 3
                    self.predicted_base_pose_t = self.get_base_pose(self._state[0:3], self._state[obj_start_idx:obj_end_idx])
                    self.obj_exist = 1
                    # print("Grasped object:", idx)
                    break
                n = n + 1

        x, y, theta = self.predicted_base_pose_t
        self._move_robot_base_to_pose(x, y, theta)

        
        self.world.step(render=self.cfg.task.mdp.render)

        # state = self._get_state()
        reward, goal_status = self._get_reward()
        state = self._get_state() # after reward beacuse part of the action is being performed in reward        
        
        # print("Reward:", reward)
        # print("State:", state)
        # print("N steps:", self._n_steps, " Episode length:", self.epsiode_length, " Action:", action)
        # print("-------------------------------------------------------------------------------------------------------------")
        # input("Press enter to continue")

        # delay for video
        # time.sleep(1)

        

        return state, reward, goal_status, {}

    
    def _get_grasp_pose(self, obj_id, obj_prim):
        grasp_offsets = None
        obj_pose_in_world = None
        robot_pose_in_world = self.ur5e_prim.get_world_pose()
        obj_pose_in_world = obj_prim.get_world_pose()

        # obj ids: cracker_box, bleach_cleanser, mustard_bottle, mug, bowl

        if obj_id == 0:
            grasp_offsets = self.cfg.grasp_poses.cracker_box.top
        elif obj_id == 1:
            grasp_offsets = self.cfg.grasp_poses.bleach_cleanser.top
        elif obj_id == 2:
            grasp_offsets = self.cfg.grasp_poses.mustard_bottle.top
        elif obj_id == 3:
            grasp_offsets = self.cfg.grasp_poses.mug.top
        elif obj_id == 4:
            grasp_offsets = self.cfg.grasp_poses.bowl.top

        grasp_offsets = self.cfg.grasp_poses.cracker_box.top

        grasp_pose_o_tran = [grasp_offsets.x_tran, grasp_offsets.y_tran, grasp_offsets.z_tran]
        grasp_pose_o_rot = Rotation.from_euler('xyz', [grasp_offsets.x_rot, grasp_offsets.y_rot, grasp_offsets.z_rot]).as_quat()
        grasp_pose_o = np.hstack((grasp_pose_o_tran, grasp_pose_o_rot))

        oTg = self._pose_to_transformation_matrix(grasp_pose_o)  

        wTo = self._isaac_pose_to_transformation_matrix(obj_pose_in_world)
        wTg = np.matmul(wTo, oTg)

        wTr = self._isaac_pose_to_transformation_matrix(robot_pose_in_world)    # ur5e base frame
        rTw = np.linalg.inv(wTr)
        rTg = np.matmul(rTw, wTg)
        quat = Rotation.from_matrix(rTg[:3,:3]).as_quat()

        grasp_pose = np.array([rTg[0,3], rTg[1,3], rTg[2,3], quat[0], quat[1], quat[2], quat[3]]) 

        return grasp_pose




    def _get_reward(self):
        reward = 0

        goal_status = False

        robot_pose = self.ur5e_prim.get_world_pose()

        # print("Objs to grasp:", self.objs_to_grasp)
        # print("Objs to grasp action:", self.objs_to_grasp_action)
        
        n = 0
        for idx, obj in enumerate(self.obj_list):
            if obj:
                if self.objs_to_grasp_action[n] == 1:
                    self.obj_list[idx] = False
                    self.obj_prims[idx].set_visibility(False)
                    self.world.step(render=self.cfg.task.mdp.render)

                    # man, nav cost
                    object_pose = self.obj_prims[idx].get_world_pose()
                    dist_to_obj = np.sqrt(np.power(object_pose[0][0] - robot_pose[0][0],2) + np.power(object_pose[0][1] - robot_pose[0][1],2))
                    curr_joint_positions = self.get_arm_joint_angles()
                            

                    # TODO As of now, we only have policy for cracker box
                    grasp_pose = self._get_grasp_pose(0, self.obj_prims[idx])
                    grasp_pose_t = self._pose_to_transformation_matrix(grasp_pose)

                    start_pos = [-0.56974803, -0.12528089,  0.40356258]
                    start_rot = [-0.65335305,  0.67443967,  0.2704206,  -0.21244674]

                    task_space_position_targets = np.array([
                        [0.3, 0.3, 0.3],  
                        [start_pos[0], start_pos[1], start_pos[2]],
                        [grasp_pose[0], grasp_pose[1], grasp_pose[2]]
                        ])
                    task_space_orientation_targets = np.array([
                        [1,0,0,0],
                        [start_rot[3], start_rot[0], start_rot[1], start_rot[2]],
                        [grasp_pose[6], grasp_pose[3], grasp_pose[4], grasp_pose[5]]
                        ])

                    t = -1
                    try:
                        trajectory = self.task_space_trajectory_generator.compute_task_space_trajectory_from_points(
                            task_space_position_targets, task_space_orientation_targets, "tool0"
                        )

                        if trajectory:
                            # print("Grasped object:", idx)
                            t = trajectory.end_time #- 4.8269
                            # print("Grasp execution time:", t)
                            # reward = reward + (50000/(1+t)) 
                            # reward = reward + 100000
                            # print("Success ", self._n_steps)

                            final_joint_config = trajectory.get_joint_targets(trajectory.end_time)[0]
                            self.robot.set_joint_positions(final_joint_config[0:6], joint_indices=np.array([0,1,2,3,4,5]))
                            self.world.step(render=self.cfg.task.mdp.render)
                    except:
                        print("Error in computing trajectory")

                    start = self.robot_init_pose_t
                    end = self.predicted_base_pose_t

                    nav_cost = self.navigation.compute_path(start, end)
                    # print("Start:", start[0:2], " Goal:", end[0:2])
                    # print("Navigation cost:", nav_cost)
                    # reward = reward +  (50000/(1+nav_cost))
                    # reward = reward +  (100000/(1+nav_cost))

                    # also change baseline reward
                    reward = - (1000 * nav_cost)

                    self.navigation_costs.append(nav_cost)
                    self.grasping_costs.append(t)
                    # np.savez('{}/data_for_evaluation_ours.npz'.format('/home/sdur'), navigation_costs=np.array(self.navigation_costs), 
                    #     grasping_costs=np.array(self.grasping_costs))

                    break
                else:
                    n = n + 1



        self._n_steps = self._n_steps + 1

        if self._n_steps == self.epsiode_length - 0:
            goal_status = True
            # print("Episode over")

        if self.cfg.task.test.save_data and goal_status:
            self.obj_status_final = np.append(self.obj_status_final, self.obj_list)
            np.savez('{}/data_for_analysis.npz'.format(self.cfg.task.test.save_dir), obj_status_init=self.obj_status_init, 
                obj_status_final=self.obj_status_final)

        # print("Reward:", reward)
        return reward, goal_status

    
    def _set_up_scene(self) -> None:
        super()._set_up_scene()
