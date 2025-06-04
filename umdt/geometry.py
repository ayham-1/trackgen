import umdt.state as state
import logging

from typing import List, Tuple, TypedDict, cast

import math
import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation

import shapely

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("umdt")


class UMDT_TrackShape(TypedDict):
    name: str
    geometries: List[List[Tuple[float, float]]]
    size: Tuple[float, float]
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    start_point_left: Tuple[float, float]
    start_point_right: Tuple[float, float]
    end_point_left: Tuple[float, float]
    end_point_right: Tuple[float, float]


class UMDT_Geometry:
    @staticmethod
    def process_track(track) -> List[UMDT_TrackShape]:
        processed_track = []
        for i in range(len(track)):
            processed_track += UMDT_Geometry.convert_track_part_to_shape(
                track[i])

        for i in range(1, len(track)):
            processed_track[i] = UMDT_Geometry.attach_track_shapes(
                processed_track[i-1], processed_track[i])
        return UMDT_Geometry.ensure_track_positive_position(processed_track)

    @staticmethod
    def convert_track_part_to_shape(track_part) -> List[UMDT_TrackShape]:
        """
            Converts from track_part with arguments to UMDT_TrackShape
            The graphical description is considered from (0,0), translation needs to be done by the caller
        """

        distance_between_cones = state.data["cfg"]["distance_between_cones"][0]
        width_of_track = state.data["cfg"]["width_of_track"][0]

        assert len(track_part)
        assert track_part[0] != "cfg"

        result = {}
        geometries = []

        track_type = track_part[0]
        track_args = track_part[1:] if len(track_part) > 1 else []
        if track_type == "start" or track_type == "end":
            length_of_part = 5
            line_left = ("line", [(0, 0), (0, length_of_part)])
            line_right = ("line", [(width_of_track, 0),
                          (width_of_track, length_of_part)])
            geometries.append(line_left)
            geometries.append(line_right)
            result = {
                "name": track_type,
                "geometries": geometries,
                "size": (width_of_track, length_of_part),
                "start_point": (width_of_track / 2, 0),
                "end_point": (width_of_track / 2, length_of_part),
                "start_point_left": (0, 0),
                "start_point_right": (width_of_track, 0),
                "end_point_left": (0, length_of_part),
                "end_point_right": (width_of_track, length_of_part),
            }
            return [UMDT_TrackShape(**result)]
        elif track_type == "straight":
            return UMDT_Geometry.convert_track_part_to_shape(
                ["curve", "right", 0.0, track_args[0]])
        elif track_type == "curve":
            dir = track_args[0]
            factor = track_args[1]
            distance = track_args[2]

            assert factor >= 0.0 and factor <= 1.0

            radius = distance / 2
            # Start and end points
            start = (0, 0)
            end = (0, distance)

            if dir == "right":
                factor *= -1

            # Intermediate control points to shape the curve
            control_point_1 = [factor * radius * (3/4), distance / 4]
            control_point_2 = [factor * radius, distance * (2/4)]
            control_point_3 = [factor * radius * (3/4), distance * (3/4)]
            control_points = [start, control_point_1,
                              control_point_2, control_point_3, end]

            geometries, x_max, y_max, x_min, y_min, start_point, end_point = UMDT_Geometry.generate_bspline(
                control_points)

            result = {
                "name": track_type,
                "geometries": geometries,
                "size": (x_max, y_max),
                "start_point": start_point,
                "end_point": end_point,
                "start_point_left": geometries[0][0],
                "start_point_right": geometries[1][0],
                "end_point_left": geometries[0][-1],
                "end_point_right": geometries[1][-1],
            }
            return [UMDT_TrackShape(**result)]
        elif track_type == "turn_left":
            return UMDT_Geometry.convert_track_part_to_shape(["curve", "left", 1.0, 20])
        elif track_type == "turn_right":
            result = UMDT_Geometry.convert_track_part_to_shape(
                ["curve", "right", 1.0, 20])
        elif track_type == "circle":
            dir = track_args[0]
            radius = track_args[1]
            degree = track_args[2]

            start = (0, 0)

            control_points = [start]

            if dir == "left":
                degree *= -1
                radius += width_of_track

            steps = 50
            for i in range(1, steps):
                angle = math.radians(i * (degree / steps))
                x = radius * math.sin(angle)
                y = radius - (radius * math.cos(angle))
                control_points.append((x, y))

            angle = math.radians(degree)
            control_points.append((radius * math.sin(angle),
                                   radius - (radius * math.cos(angle))))

            geometries, x_max, y_max, x_min, y_min, start_point, end_point = UMDT_Geometry.generate_bspline(
                control_points)

            result = {
                "name": track_type,
                "geometries": geometries,
                "size": (x_max, y_max),
                "start_point": start_point,
                "end_point": end_point,
                "start_point_left": geometries[0][0],
                "start_point_right": geometries[1][0],
                "end_point_left": geometries[0][-1],
                "end_point_right": geometries[1][-1],
            }
            return [UMDT_TrackShape(**result)]
        elif track_type == "hairpin":
            return UMDT_Geometry.convert_track_part_to_shape(["circle", "right", width_of_track * 1.5, 180])
        elif track_type == "esses":
            dir = track_args[0]
            follow_prev_straight = track_args[1]
            number_of_hcircles = track_args[2]

            parts = []
            if follow_prev_straight:
                parts.append(["circle", dir, width_of_track * 2, 90])
                dir = "left" if dir == "right" else "right"

            for i in range(number_of_hcircles):
                parts.append(["circle", dir, width_of_track * 2, 180])
                dir = "left" if dir == "right" else "right"

            if follow_prev_straight:
                parts.append(["circle", dir, width_of_track * 2, 90])
                dir = "left" if dir == "right" else "right"

            return UMDT_Geometry.process_track(parts)
        elif track_type == "chicane":
            dir = track_args[0]
            degree1 = track_args[1]
            distance = track_args[2]
            degree2 = track_args[3]

            track_segment = []
            track_segment.append(["curve", dir, 0.3, width_of_track * 3])
            track_segment.append(["straight", distance])
            dir = "left" if dir == "right" else "right"
            track_segment.append(["curve", dir, 0.3, width_of_track * 3])

            return UMDT_Geometry.process_track(track_segment)
        elif track_type == "double_apex_turn":
            dir = track_args[0]
            distance = track_args[1]

            track_segment = []
            track_segment.append(["circle", dir, width_of_track * 2, 90])
            track_segment.append(["straight", distance])
            track_segment.append(["circle", dir, width_of_track * 2, 90])

            return UMDT_Geometry.process_track(track_segment)

        raise NotImplementedError(f"Track part {track_part} not implemented")

    @staticmethod
    def attach_track_shapes(rg1, rg2):
        """
        rg1 and rg2 are results from calling convert_track_part_to_shape()

        return rg2 translated to start and end at rg1
        """

        # find translation vector
        translation = (rg1["end_point"][0] - rg2["start_point"][0],
                       rg1["end_point"][1] - rg2["start_point"][1])

        # translate geometries
        for i in range(len(rg2["geometries"])):
            rg2["geometries"][i] = [
                (x + translation[0], y + translation[1]) for x, y in rg2["geometries"][i]]

        # update start and end points
        rg2["start_point"] = (rg2["start_point"][0] + translation[0],
                              rg2["start_point"][1] + translation[1])
        rg2["end_point"] = (rg2["end_point"][0] + translation[0],
                            rg2["end_point"][1] + translation[1])

        # update left and right start and end points
        rg2["start_point_left"] = (rg2["start_point_left"][0] + translation[0],
                                   rg2["start_point_left"][1] + translation[1])
        rg2["start_point_right"] = (rg2["start_point_right"][0] + translation[0],
                                    rg2["start_point_right"][1] + translation[1])
        rg2["end_point_left"] = (rg2["end_point_left"][0] + translation[0],
                                 rg2["end_point_left"][1] + translation[1])
        rg2["end_point_right"] = (rg2["end_point_right"][0] + translation[0],
                                  rg2["end_point_right"][1] + translation[1])

        # find rotation between end of rg1 and start of rg2 so that rg2 is aligned
        # with rg1
        vec_point_left1 = np.array(
            rg2["start_point_left"]) - np.array(rg2["start_point"])
        vec_point_left2 = np.array(
            rg1["end_point_left"]) - np.array(rg2["start_point"])
        # use cross product mathematics to find angle between two vectors
        cross_product = np.dot(vec_point_left1, vec_point_left2)
        ratio = max(-1, min(1, cross_product /
                    (np.linalg.norm(vec_point_left1) * np.linalg.norm(vec_point_left2))))
        angle = math.degrees(math.acos(ratio))
        # Determinant for sign
        sign = vec_point_left1[0] * vec_point_left2[1] - \
            vec_point_left1[1] * vec_point_left2[0]
        signed_angle = angle if sign >= 0 else -angle

        # rotate geometries
        r = Rotation.from_euler("z", signed_angle, degrees=True)

        def rotate_points(points, pivot=rg2["start_point"]):
            translated_points = points - np.array(pivot)
            rotated_points = r.apply(
                np.c_[translated_points, np.zeros(len(points))])[:, :2]
            return rotated_points + np.array(pivot)

        for i in range(len(rg2["geometries"])):
            points = rg2["geometries"][i]
            rg2["geometries"][i] = rotate_points(points)

        # rotate start and end points
        rg2["start_point"] = rotate_points([rg2["start_point"]])[0]
        rg2["end_point"] = rotate_points([rg2["end_point"]])[0]

        # rotate left and right start and end points
        rg2["start_point_left"] = rotate_points([rg2["start_point_left"]])[0]
        rg2["start_point_right"] = rotate_points([rg2["start_point_right"]])[0]
        rg2["end_point_left"] = rotate_points([rg2["end_point_left"]])[0]
        rg2["end_point_right"] = rotate_points([rg2["end_point_right"]])[0]

        return rg2

    @staticmethod
    def generate_circle(radius, degree):
        control_points = []
        steps = 100
        for i in range(1, steps):
            angle = math.radians(i * (degree / steps))
            x = radius * math.sin(angle)
            y = radius - (radius * math.cos(angle))
            control_points.append((x, y))

        angle = math.radians(degree)
        control_points.append((radius * math.sin(angle),
                               radius - (radius * math.cos(angle))))

        return control_points

    @staticmethod
    def generate_bspline(control_points):
        width_of_track = state.data["cfg"]["width_of_track"][0]

        k = 3
        control_x, control_y = zip(*control_points)

        # Define knots (clamped uniform knots)
        n_control_points = len(control_points)
        n_knots = n_control_points + k + 1
        knots = np.concatenate(
            ([0]*k, np.linspace(0, 1, n_knots - 2*k), [1]*k))

        # Create separate BSplines for x and y
        b_spline_x = interpolate.BSpline(
            knots, control_x, k, extrapolate=False)
        b_spline_y = interpolate.BSpline(
            knots, control_y, k, extrapolate=False)

        # Evaluate the spline on a parameter grid
        u = np.linspace(0, 1, 100)  # Parameter values
        x_spline = b_spline_x(u)
        y_spline = b_spline_y(u)

        dx = b_spline_x.derivative()(u)
        dy = b_spline_y.derivative()(u)

        # Compute the normals
        normals = np.array([-1 * dy, dx])
        normals /= np.linalg.norm(normals, axis=0)

        x_spline = x_spline - min(x_spline)
        y_spline = y_spline - min(y_spline)

        # Offset the points by multiple of normal of each point
        x_spline_offset = x_spline - width_of_track * normals[0]
        y_spline_offset = y_spline - width_of_track * normals[1]

        # move splines so that negative values are positive
        offset = (min(x_spline_offset), min(y_spline_offset))
        if offset[0] < 0 or offset[1] < 0:
            x_spline_offset = x_spline_offset - offset[0]
            x_spline = x_spline - offset[0]
            y_spline_offset = y_spline_offset - offset[1]
            y_spline = y_spline - offset[1]

        spline = np.array(list(zip(x_spline, y_spline)))
        spline_offset = np.array(
            list(zip(x_spline_offset, y_spline_offset)))

        # determine min y-value
        min_y = min(y_spline[np.argmin(y_spline)],
                    y_spline_offset[np.argmin(y_spline_offset)])
        if min_y < 0:
            spline = [(x, y - min_y) for x, y in spline]
            spline_offset = [(x, y - min_y) for x, y in spline_offset]

        # determine min x-value
        min_x = min(x_spline[np.argmin(x_spline)],
                    x_spline_offset[np.argmin(x_spline_offset)])
        if min_x < 0:
            spline = [(x - min_x, y) for x, y in spline]
            spline_offset = [(x - min_x, y) for x, y in spline_offset]

        # determine max values
        spline_max = np.max(spline, axis=0)
        spline_offset_max = np.max(spline_offset, axis=0)

        spline_points, offset_spline_points, x_max, y_max = (spline, spline_offset,
                                                             max(spline_offset_max[0], spline_max[0]), max(spline_offset_max[1], spline_max[1]))

        # get point between starting points
        def get_mid_point(p1, p2):
            m = (p1[1] - p2[1]) /  \
                (p1[0] - p2[0])
            c = p1[1] - m * p1[0]
            x_mid = (p1[0] + p2[0]) / 2
            y_mid = m * x_mid + c

            return (x_mid, y_mid)

        start_point = get_mid_point(
            spline_points[0], offset_spline_points[0])
        end_point = get_mid_point(
            spline_points[-1], offset_spline_points[-1])

        return [spline_points, offset_spline_points], x_max, y_max, min_x, min_y, start_point, end_point

    @staticmethod
    def ensure_track_positive_position(track):
        min_x = 0
        min_y = 0
        for rgeometries in track:
            for geometry in rgeometries["geometries"]:
                min_x = min(min_x, min([x for x, _ in geometry]))
                min_y = min(min_y, min([y for _, y in geometry]))

        updated_track = []
        for rgeometries in track:
            updated_rgeometries = rgeometries.copy()
            updated_rgeometries["geometries"] = []
            for geometry in rgeometries["geometries"]:
                if min_x < 0:
                    geometry = [(x + -min_x, y) for x, y in geometry]
                if min_y < 0:
                    geometry = [(x, y + -min_y) for x, y in geometry]

                updated_rgeometries["geometries"].append(geometry)

            # update start and end points
            updated_rgeometries["start_point"] = (rgeometries["start_point"][0] + -min_x,
                                                  rgeometries["start_point"][1] + -min_y)
            updated_rgeometries["end_point"] = (rgeometries["end_point"][0] + -min_x,
                                                rgeometries["end_point"][1] + -min_y)

            # update left and right start and end points
            updated_rgeometries["start_point_left"] = (rgeometries["start_point_left"][0] + -min_x,
                                                       rgeometries["start_point_left"][1] + -min_y)
            updated_rgeometries["start_point_right"] = (rgeometries["start_point_right"][0] + -min_x,
                                                        rgeometries["start_point_right"][1] + -min_y)
            updated_rgeometries["end_point_left"] = (rgeometries["end_point_left"][0] + -min_x,
                                                     rgeometries["end_point_left"][1] + -min_y)
            updated_rgeometries["end_point_right"] = (rgeometries["end_point_right"][0] + -min_x,
                                                      rgeometries["end_point_right"][1] + -min_y)

            updated_track.append(updated_rgeometries)

        return updated_track

    @staticmethod
    def get_track_size(track):
        min_x = 0
        min_y = 0
        max_x = 0
        max_y = 0
        for rgeometries in track:
            for geometry in rgeometries["geometries"]:
                min_x = min(min_x, min([x for x, _ in geometry]))
                min_y = min(min_y, min([y for _, y in geometry]))

                max_x = max(max_x, max([x for x, _ in geometry]))
                max_y = max(max_y, max([y for _, y in geometry]))

        return (max_x - min_x, max_y - min_y)

    @staticmethod
    def discard_overlaps(track, intersections):
        new_track = []

        def segment_geometry(geometry):
            segments_discarded = []
            segments = []
            segment_current = []
            discarding = False
            for i in range(1, len(geometry)):
                points = geometry
                for intersect in intersections:
                    intersection_point = shapely.Point(intersect)
                    buffer_radius = 1e-3
                    buffered_point = intersection_point.buffer(buffer_radius)
                    if buffered_point.intersects(shapely.LineString([points[i-1], points[i]])):
                        segment_current.append(intersect)
                        if discarding:
                            segments_discarded.append(segment_current)
                        else:
                            segments.append(segment_current)
                        discarding = not discarding
                        segment_current = [intersect]
                        break

                if i == 1:
                    segment_current.append(points[0])
                segment_current.append(points[i])

            if discarding:
                segments_discarded.append(segment_current)
            else:
                segments.append(segment_current)

            return (segments, segments_discarded)

        for part in track:
            geometries = part["geometries"]
            new_geometries = []

            for i in range(len(geometries)):
                geometry = geometries[i]
                segments, segments_discarded = segment_geometry(geometry)

                if len(segments) == 1 and len(segments_discarded) == 0:
                    new_geometries += segments
                if len(segments) == 0 and len(segments_discarded) == 1:
                    new_geometries += segments_discarded
                else:
                    for j in range(0, len(segments) - 1):
                        if len(segments_discarded[j]) < len(segments[j]) \
                                and len(segments_discarded[j]) < len(segments[j + 1]):
                            new_geometries.append(segments[j])
                            new_geometries.append(segments[j + 1])
                        else:
                            if len(segments_discarded[j]) > len(segments[j]):
                                new_geometries.append(segments_discarded[j])

                            if len(segments_discarded[j]) < len(segments[j]):
                                new_geometries.append(segments[j + 1])

            part["geometries"] = new_geometries
            new_track.append(part)

        return new_track

    @staticmethod
    def get_intersections(track):
        lines = []
        for part in track:
            for geometry in part["geometries"]:
                lines.append(shapely.LineString(geometry))

        result = []
        for line in lines:
            for other_line in lines:
                if line.equals(other_line):
                    continue

                intersection = line.intersection(other_line)
                coords = []
                if intersection.is_empty:
                    continue
                if intersection.geom_type == 'Point':
                    # print(f"Intersection Point: {intersection}")
                    coords += list(intersection.coords)
                elif intersection.geom_type == 'MultiPoint':
                    # print(f"Multi Intersection Point: {intersection}")
                    geoms = list(intersection.geoms)
                    for geom in geoms:
                        coords += list(geom.coords)

                if len(coords):
                    result += coords

        # filter incorrect intersections
        result = list(set(result))
        print(result)
        for part in track:
            for intersect in result:
                buffer_radius = 1e-2
                buffered_point = shapely.Point(intersect).buffer(buffer_radius)
                if shapely.Point(part["start_point_left"]).intersects(buffered_point):
                    result = list(filter((intersect).__ne__, result))
                    continue
                if shapely.Point(part["end_point_left"]).intersects(buffered_point):
                    result = list(filter((intersect).__ne__, result))
                    continue
                if shapely.Point(part["start_point_right"]).intersects(buffered_point):
                    result = list(filter((intersect).__ne__, result))
                    continue
                if shapely.Point(part["end_point_right"]).intersects(buffered_point):
                    result = list(filter((intersect).__ne__, result))
                    continue

        return list(set(result))
