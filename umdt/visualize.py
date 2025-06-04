import umdt.state as state
import logging

from umdt.geometry import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("umdt")


class UMDT_Visualize:
    @staticmethod
    def visualize(parsed):
        """
        Uses pygame to visualize the track described in path.
        """
        import pygame

        pygame.init()
        pygame.display.set_caption("UMDT Visualizer")

        screen_info = pygame.display.Info()
        screen_width, screen_height = screen_info.current_w, screen_info.current_h

        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 3)
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
        screen_width, screen_height = screen.get_size()
        clock = pygame.time.Clock()
        running = True

        camera_offset = pygame.Vector2(screen_width / 2, screen_height / 2)
        # camera_offset = pygame.Vector2(0, 0)
        camera_speed = 5 * state.sf * 0.5

        track = [["circle", "right", 20, 180], ["turn_right"]]
        # track = [["circle", "right", 20, 180]]
        track = [["curve", "right", 0.5, 20], ["curve", "right", 0.75, 20], [
            "curve", "right", 1.0, 20], ["curve", "right", 1.0, 20], ["straight", 20]]

        track = [["straight", 30], [
            "circle", "right", 20, 270], ["straight", 50]]
        track = [["circle", "left", 20, 360], ["circle", "right", 20, 360]]
        track = parsed["parts"]
        track = [["circle", "right", 20, 90], ["circle", "left", 20, 180], [
            "circle", "right", 20, 180], ["circle", "left", 20, 180]]
        track = [["esses", "right", True, 5]]
        track = [["chicane", "right", 45, 10, 45]]
        track = [["double_apex_turn", "left", 10]]

        track = UMDT_Geometry.process_track(track)

        intersections = UMDT_Geometry.get_intersections(track)
        track = UMDT_Geometry.discard_overlaps(track, intersections)

        size = UMDT_Geometry.get_track_size(track)
        surface = pygame.Surface(
            (size[0] * state.sf + 20, size[1] * state.sf + 20))
        index_of_focused_part = 0

        def focus_camera_on_part(index, start_point=True):
            point = track[index]["start_point"]
            if not start_point:
                point = track[index]["end_point"]
            point = (point[0] * state.sf,
                     point[1] * state.sf)
            camera_offset.x = (screen_width / 2) - point[0]
            camera_offset.y = (screen_height / 2) - point[1]

        focus_camera_on_part(index_of_focused_part)

        font = pygame.font.Font(None, 24)  # Use default font, size 36

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if pygame.key.name(event.key) == "]":
                        index_of_focused_part += 1
                        if index_of_focused_part >= len(track):
                            index_of_focused_part = 0
                        focus_camera_on_part(
                            index_of_focused_part, not event.mod & pygame.KMOD_SHIFT)
                    if pygame.key.name(event.key) == "[":
                        index_of_focused_part -= 1
                        if index_of_focused_part < 0:
                            index_of_focused_part = len(track) - 1
                        focus_camera_on_part(
                            index_of_focused_part, not event.mod & pygame.KMOD_SHIFT)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                pygame.quit()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                camera_offset.x += camera_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                camera_offset.x -= camera_speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                camera_offset.y += camera_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                camera_offset.y -= camera_speed
            if keys[pygame.K_MINUS]:
                state.sf = max(0, state.sf - 0.10)
                surface = pygame.Surface(
                    (size[0] * state.sf + 20, size[1] * state.sf + 20))
            if (keys[pygame.K_EQUALS] and pygame.key.get_mods() & pygame.KMOD_SHIFT) or keys[pygame.K_EQUALS]:
                state.sf = min(30, state.sf + 0.10)
                surface = pygame.Surface(
                    (size[0] * state.sf + 20, size[1] * state.sf + 20))
            if keys[pygame.K_0]:
                index_of_focused_part = 0
                focus_camera_on_part(index_of_focused_part)
                state.sf = 15

            screen.fill("black")
            surface.fill("black")

            for part in track:
                # draw start and end points of track part
                pygame.draw.circle(surface, "red", (part["start_point"][0] * state.sf,
                                                    part["start_point"][1] * state.sf), 2, 2)
                pygame.draw.circle(surface, "red", (part["end_point"][0] * state.sf,
                                                    part["end_point"][1] * state.sf), 2, 2)

                # draw left and right start and end points
                pygame.draw.circle(surface, "green", (part["start_point_left"][0] * state.sf,
                                                      part["start_point_left"][1] * state.sf), 2, 2)
                pygame.draw.circle(surface, "yellow", (part["start_point_right"][0] * state.sf,
                                                       part["start_point_right"][1] * state.sf), 2, 2)
                pygame.draw.circle(surface, "green", (part["end_point_left"][0] * state.sf,
                                                      part["end_point_left"][1] * state.sf), 2, 2)
                pygame.draw.circle(surface, "yellow", (part["end_point_right"][0] * state.sf,
                                                       part["end_point_right"][1] * state.sf), 2, 2)

                for geometry in part["geometries"]:
                    # draw geometry
                    points = [(x[0] * state.sf, x[1] * state.sf)
                              for x in geometry]
                    pygame.draw.aalines(
                        surface, "white", False, points)

            # render intersectiosn
            for intersection in intersections:
                pygame.draw.circle(surface, "cyan", (intersection[0] * state.sf,
                                                     intersection[1] * state.sf), 2, 2)

            screen.blit(surface, camera_offset)
            # draw center cross
            cross_width = 15
            cross = pygame.Surface(
                (cross_width * 2, cross_width * 2), pygame.SRCALPHA)
            pygame.draw.aaline(cross, "orange", (0, cross_width),
                               (cross_width * 2, cross_width))
            pygame.draw.aaline(cross, "orange", (cross_width, 0),
                               (cross_width, cross_width * 2))
            screen.blit(cross, (screen_width / 2 - cross_width,
                        screen_height / 2 - cross_width))

            # fps
            fps = clock.get_fps()
            fps_text = font.render(
                f"FPS: {int(fps)}", True, (255, 255, 255))
            screen.blit(fps_text, (10, 10))

            # position of selected part
            part_index_text = font.render(
                f"Part: {index_of_focused_part}", True, (255, 255, 255))
            part_start_text = font.render(
                f"Start: {(int(track[index_of_focused_part]['start_point'][0]), int(track[index_of_focused_part]['start_point'][1]))}", True, (255, 255, 255))
            part_end_text = font.render(
                f"End: {(int(track[index_of_focused_part]['end_point'][0]), int(track[index_of_focused_part]['end_point'][1]))}", True, (255, 255, 255))

            camera_offset_text = font.render(
                f"Camera offset: {(int(camera_offset[0]), int(camera_offset[1]))}", True, (255, 255, 255))

            screen.blit(part_index_text, (11, 30))
            screen.blit(part_start_text, (11, 50))
            screen.blit(part_end_text, (11, 70))
            screen.blit(camera_offset_text, (11, 90))

            pygame.display.flip()
            pygame.display.update()
            clock.tick(30)

        pygame.quit()
