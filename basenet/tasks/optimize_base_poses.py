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


class OptimizeBasePoses(Task):
    """
    Optimize base poses
    """

    def __init__(
        self, cfg
    ) -> None:
        if cfg.task.test.save_data:
            self.obj_poses = []
            self.robot_poses = []
            self.table_poses = []

        super().__init__(cfg)

        self.navigation = Navigation(cfg)

        self.robot_init_pose_t = None
        self.predicted_base_pose_t = None
        


    def reset(self, state=None):
        self.predicted_base_pose_t = None
        self.robot_init_pose_t = None
        state = super().reset()
        self.objs_to_grasp = self.obj_list
        self._n_steps = 0


        self.robot_init_pose_t = self.get_2D_robot_pose_in_table_frame()        
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


    def step(self, action):
        self.robot_init_pose_t = self.get_2D_robot_pose_in_table_frame() 
        # print("Action in step:", action)

        x = action[0]
        y = action[1]
        theta = action[2]


        # for relative frame
        # T_oa = np.eye(4)
        # T_oa[:3,:3] = Rotation.from_euler('z', theta).as_matrix()
        # T_oa[0,3] = x
        # T_oa[1,3] = y

        # T_to = np.eye(4)
        # T_to[:3,:3] = Rotation.from_euler('z', self._state[2]).as_matrix()
        # T_to[0,3] = self._state[0]
        # T_to[1,3] = self._state[1]

        # T_ta = np.matmul(T_to, T_oa)
        # x = T_ta[0,3]
        # y = T_ta[1,3]
        # theta = Rotation.from_matrix(T_ta[:3,:3]).as_euler('xyz')[2]

        self.predicted_base_pose_t = np.array([x, y, theta])

        # print("Predicted base pose (in world):", x, y, theta)

        # print("Action in step:", action)

        # most probably this is not needed as world and table only defers in Z
        # [x,y,theta] = self.table_to_world_frame(x,y,theta)

        # input("Not move yet {}".format(self.robot_base_prim.get_world_pose()[0]))
        self._move_robot_base_to_pose(x, y, theta)
        
        # Add code here
        
        self.world.step(render=self.cfg.task.mdp.render)
        # input("Moved now {}".format(self.robot_base_prim.get_world_pose()[0]))

        state = self._get_state()
        reward, goal_status = self._get_reward()
        
        
        self._n_steps = self._n_steps + 1

        # delay for video
        # time.sleep(1)

        # input("Press enter to continue")

        return state, reward, goal_status, {}

    
    def _get_grasp_pose(self, obj_id):
        grasp_offsets = None
        obj_pose_in_world = None
        robot_pose_in_world = self.ur5e_prim.get_world_pose()

        # obj ids: cracker_box, bleach_cleanser, mustard_bottle, mug, bowl

        if obj_id == 0:
            grasp_offsets = self.cfg.grasp_poses.cracker_box.top
            obj_pose_in_world = self.cracker_box_prim.get_world_pose()
        elif obj_id == 1:
            grasp_offsets = self.cfg.grasp_poses.bleach_cleanser.top
            obj_pose_in_world = self.bleach_cleanser_prim.get_world_pose()
        elif obj_id == 2:
            grasp_offsets = self.cfg.grasp_poses.mustard_bottle.top
            obj_pose_in_world = self.mustard_bottle_prim.get_world_pose()
        elif obj_id == 3:
            grasp_offsets = self.cfg.grasp_poses.mug.top
            obj_pose_in_world = self.mug_prim.get_world_pose()
        elif obj_id == 4:
            grasp_offsets = self.cfg.grasp_poses.bowl.top
            obj_pose_in_world = self.bowl_prim.get_world_pose()


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

        # since we have only 1 step for now
        goal_status = False

        if self.check_collision():
            reward = -20000
            goal_status = True
            return reward, goal_status


        object_pose = self.cracker_box_prim.get_world_pose()
        robot_pose = self.ur5e_prim.get_world_pose()

        dist_to_obj = np.sqrt(np.power(object_pose[0][0] - robot_pose[0][0],2) + 
                        np.power(object_pose[0][1] - robot_pose[0][1],2))

        curr_joint_positions = self.get_arm_joint_angles()

        for idx, status in enumerate(self.objs_to_grasp):
            # Overidden for experiment
            if status and idx == 0:
                grasp_pose = self._get_grasp_pose(idx)
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

                try:
                    trajectory = self.task_space_trajectory_generator.compute_task_space_trajectory_from_points(
                        task_space_position_targets, task_space_orientation_targets, "tool0"
                    )

                    if trajectory:
                        t = trajectory.end_time
                        reward = reward + (50000/(1+t)) + 100000
                        goal_status = True

                        # final_joint_config = trajectory.get_joint_targets(trajectory.end_time)[0]
                        # self.robot.set_joint_positions(final_joint_config[0:6], joint_indices=np.array([0,1,2,3,4,5]))
                        # self.world.step(render=self.cfg.task.mdp.render)

                        if self.cfg.task.mdp.stage == 2:
                            start = self.robot_init_pose_t
                            end = self.predicted_base_pose_t

                            nav_cost = self.navigation.compute_path(start, end)
                            reward = reward +  (50000/(1+nav_cost))
                    else:
                        reward = reward - 10000
                except:
                    reward = reward + 000
                    print("Error in computing trajectory")

        if self._n_steps > self.cfg.task.mdp.horizon:
            goal_status = True
        
        return reward, goal_status

    
    def _set_up_scene(self) -> None:
        super()._set_up_scene()
