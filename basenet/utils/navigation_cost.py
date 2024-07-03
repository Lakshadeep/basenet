import numpy as np
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from shapely.geometry import Polygon
import math
import networkx as nx

from math import pi
from shapely.ops import linemerge, unary_union, polygonize
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import split

import matplotlib.cm as cm
import matplotlib
from tqdm import tqdm


class Navigation():
    """
    Computes navigation cost
    """

    def __init__(
        self ,
        cfg
    ) -> None:
        self.cfg = cfg
        self.table_corners = self._get_table_corners()
        self.table_polygon = Polygon(self.table_corners)
        self.turn_points = self._get_turn_points()


    def _get_table_corners(self, full=False):
        tl = abs(self.cfg.task.mdp.table_dimensions.y_max)
        tw = abs(self.cfg.task.mdp.table_dimensions.x_max)
        t = pow(pow(tl,2) + pow(tw,2), 0.5)
        ang = math.atan(tl/tw)
        table_c1 = [0 + (t*np.cos(self._wrap_angle(0 + ang))), 0 + (t*np.sin(self._wrap_angle(0 + ang))) ]
        table_c2 = [0 + (t*np.cos(self._wrap_angle(0 - ang))), 0 + (t*np.sin(self._wrap_angle(0 - ang))) ]
        table_c3 = [0 + (t*np.cos(self._wrap_angle(0 + ang - np.pi))), 0 + (t*np.sin(self._wrap_angle(0 + ang - np.pi)))]
        table_c4 = [0 + (t*np.cos(self._wrap_angle(0 - ang + np.pi))), 0 + (t*np.sin(self._wrap_angle(0 - ang + np.pi)))]
        
        table = np.array([table_c1, table_c2, table_c3, table_c4])
        if full:
            table = np.array([table_c1, table_c2, table_c3, table_c4, table_c1])
        return table

    def _wrap_angle(self, angle):
        angle = math.fmod(angle, 2 * np.pi)
        if (angle >= np.pi):
            angle -= 2 * np.pi
        elif (angle <= -np.pi):
            angle += 2 * np.pi
        return angle

    def _get_turn_points(self):
        tl = abs(self.cfg.task.mdp.table_dimensions.y_max)
        tw = abs(self.cfg.task.mdp.table_dimensions.x_max)
        turn_points = np.array([[tw+0.3, tl+0.45], [-tw-0.3, tl+0.45], [-tw-0.3, -tl-0.45], [tw+0.3, -tl-0.45]])
        return turn_points


    def _orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if val == 0 else 1 if val > 0 else 2


    def _on_segment(self, p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


    def _do_segments_intersect(self, p1, q1, p2, q2):
        o1 = self._orientation(p1, q1, p2)
        o2 = self._orientation(p1, q1, q2)
        o3 = self._orientation(p2, q2, p1)
        o4 = self._orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self._on_segment(p1, p2, q1):
            return True
        if o2 == 0 and self._on_segment(p1, q2, q1):
            return True
        if o3 == 0 and self._on_segment(p2, p1, q2):
            return True
        if o4 == 0 and self._on_segment(p2, q1, q2):
            return True

        return False


    def _is_visible(self, point1, point2, rectangle_corners):
        for i in range(len(rectangle_corners)):
            start_point = rectangle_corners[i]
            end_point = rectangle_corners[(i + 1) % len(rectangle_corners)]

            # Check if the line segment between point1 and point2 intersects with any rectangle edge
            if self._do_segments_intersect(point1, point2, start_point, end_point):
                return False
        return True


    def _build_visibility_graph(self, rectangle_corners, turn_points, outside_point1, outside_point2):
        G = nx.Graph()

        # Add vertices for the rectangle corners and outside points
        all_points = np.vstack((turn_points, outside_point1, outside_point2))
        for i, point in enumerate(all_points):
            G.add_node(i, pos=tuple(point))

        # Add edges for visible pairs of points
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                point1, point2 = all_points[i], all_points[j]
                if self._is_visible(point1, point2, rectangle_corners):
                    # print("Connection ", i, j)
                    G.add_edge(i, j, weight=np.linalg.norm(point1 - point2))
        return G

    def _find_shortest_path_outside_rectangle(self, rectangle_corners, turn_points, outside_point1, outside_point2):
        G = self._build_visibility_graph(rectangle_corners, turn_points, outside_point1, outside_point2)

        # Find the shortest path using Dijkstra's algorithm
        start_node = len(rectangle_corners)
        end_node = len(rectangle_corners) + 1
        
        cost = 50
        path_coordinates = None

        try:
            shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
            cost = nx.path_weight(G, shortest_path, weight='weight')
            # Extract the actual coordinates of the points in the shortest path
            path_coordinates = [G.nodes[node]['pos'] for node in shortest_path]
        except:
            pass
            
        
        # print("Shortest path cost:", cost)

        return cost, path_coordinates

    def _wrap_angle(self, angle):
        angle = math.fmod(angle, 2 * np.pi)
        if (angle >= np.pi):
            angle -= 2 * np.pi
        elif (angle <= -np.pi):
            angle += 2 * np.pi
        return angle

    # def compute_angular_cost(self, start, goal, t_path):
    #     total_ang = 0
    #     prev_angle = start[2]

    #     # print("Start angle:", np.rad2deg(start[2]))
    #     # print("Goal angle:", np.rad2deg(goal[2]))
    #     if len(t_path) > 1: # and len(t_path) < 3:
    #         for i in range(1, len(t_path)):
    #             y_diff = t_path[i][1] - t_path[i-1][1]
    #             x_diff = t_path[i][0] - t_path[i-1][0]
    #             angle1 =  math.atan2(y_diff, x_diff)
    #             angle2 = self._wrap_angle(angle1 + np.pi)

    #             print("Angle 1", np.rad2deg(angle1))
    #             print("Angle 2", np.rad2deg(angle2))

    #             angle1_diff = abs(self._wrap_angle(angle1 - prev_angle))
    #             angle2_diff = abs(self._wrap_angle(angle2 - prev_angle))

    #             print("Angle 1 diff:", np.rad2deg(angle1_diff))
    #             print("Angle 2 diff:", np.rad2deg(angle2_diff))

    #             if angle1_diff < angle2_diff:
    #                 total_ang = total_ang + angle1_diff
    #                 prev_angle = angle1
    #             else:
    #                 total_ang = total_ang + angle2_diff
    #                 prev_angle = angle2

    #             print("Total angle:", np.rad2deg(total_ang), i)

        
    #     print("Final turn angle:", np.rad2deg(self._wrap_angle(goal[2] - prev_angle)))
    #     total_ang = total_ang + abs(self._wrap_angle(goal[2] - prev_angle))
    #     print("Total angular cost:", np.rad2deg(total_ang))
    #     return total_ang

    def compute_angular_cost(self, start, goal, t_path):
        total_ang1 = 0
        total_ang2 = 0
        
        prev_angle = start[2]

        if len(t_path) > 1:
            for i in range(1, len(t_path)):
                y_diff = t_path[i][1] - t_path[i-1][1]
                x_diff = t_path[i][0] - t_path[i-1][0]
                angle =  math.atan2(y_diff, x_diff)

                angle_diff = abs(self._wrap_angle(angle - prev_angle))

                # print("Angle 1 diff:", np.rad2deg(angle_diff))

                total_ang1 = total_ang1 + angle_diff
                prev_angle = angle
            
                # print("Total angle 1:", np.rad2deg(total_ang1), i)

        total_ang1 = total_ang1 + abs(self._wrap_angle(goal[2] - prev_angle))
        # print("Total angular cost 1:", np.rad2deg(total_ang1))

        prev_angle = start[2]

        if len(t_path) > 1: # and len(t_path) < 3:
            for i in range(1, len(t_path)):
                y_diff = t_path[i][1] - t_path[i-1][1]
                x_diff = t_path[i][0] - t_path[i-1][0]
                angle =  math.atan2(y_diff, x_diff)
                angle = self._wrap_angle(angle + np.pi)

                angle_diff = abs(self._wrap_angle(angle - prev_angle))

                # print("Angle diff:", np.rad2deg(angle1_diff))

                total_ang2 = total_ang2 + angle_diff
                prev_angle = angle

                # print("Total angle 2:", np.rad2deg(total_ang2), i)

        total_ang2 = total_ang2 + abs(self._wrap_angle(goal[2] - prev_angle))
        # print("Total angular cost 2:", np.rad2deg(total_ang2))
        
        return np.min(np.array([total_ang1, total_ang2]))


    def compute_path(self, start, goal, visualize=False):
        start_t = start[0:2]
        goal_t = goal[0:2]
        tran_dist, path_coordinates = self._find_shortest_path_outside_rectangle(self.table_corners, self.turn_points, start_t, goal_t)

        # print("Path coordinates:", path_coordinates)
        
        tran_cost = tran_dist/self.cfg.task.mdp.navigation.max_tran_speed

        # print("Tran cost:", tran_cost)

        rot_cost = 50
        if path_coordinates:
            rot_dist = self.compute_angular_cost(start, goal, path_coordinates)
            rot_cost = abs(rot_dist/self.cfg.task.mdp.navigation.max_rot_speed)
            # print("Rot cost:", rot_cost)
        
        if visualize:
            plt.figure()
            plt.plot(*self.table_corners.T, 'ro-', label='Rectangle Corners')
            plt.plot(*start, 'go', label='Start')
            plt.plot(*goal, 'bo', label='Goal')
            plt.plot(*np.array(path_coordinates).T, 'k-', label='Shortest Path')
            plt.legend()
            plt.title('Shortest Path')
            plt.xlim([-2.0,2.0])
            plt.ylim([-2.0,2.0])
            # plt.axis('equal')
            plt.savefig('/home/sdur/nav_cost.png')

        
        cost = tran_cost + rot_cost
        # print("Total nav cost:", cost)
        return cost 