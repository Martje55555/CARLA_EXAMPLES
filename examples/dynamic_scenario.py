from ast import Lambda
import glob
from multiprocessing.context import set_spawning_popen
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3 / 255.0


def main():
    # list to store all actors to destroy together after simulation
    actor_list = []

    try:
        # client will be accepting requests from localhost/2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)

        # get world that is currently running
        world = client.get_world()

        # get the list of blueprints that is required to spawn actors
        #settings = world.get_settings()


        # must be less than 0.1, or else physics will be noisy
        # settings.fixed_delta_seconds = 0.05
        # must use fixed delta seconds and synchronous mode for python api controlled sim, or else
        # camera and sensor data may not match simulation properly and will be noisy
        #settings.synchronous_mode = True
        #world.apply_settings(settings)

        weather = carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=100.0,
            sun_altitude_angle=110.0)

        #or use precomputed weathers
        #weather = carla.WeatherParameters.WetCloudySunset

        world.set_weather(weather)

        blueprint_library = world.get_blueprint_library()

        # get random vehicle from blueprint
        bp = blueprint_library.filter('model3')[0]

        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())
        # spawn the vehicle
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # set vehicle to autopilot
        vehicle.set_autopilot(True)

         # add more vehicles to the world
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 30):
            transform.location.x += 8.0

            bp = random.choice(blueprint_library.filter('vehicle'))

            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)

        # attatch camera to vehicle
        vehicle_cam = blueprint_library.find('sensor.camera.rgb')
        # vehicle_cam.set_attribute('PostProcessing', 'SceneFinal')
        vehicle_cam.set_attribute('image_size_x', f"{IM_WIDTH}")
        vehicle_cam.set_attribute('image_size_y', f"{IM_HEIGHT}")
        vehicle_cam.set_attribute('fov', '110')

        # set camere near hood of vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        sensor = world.spawn_actor(vehicle_cam, spawn_point, attach_to=vehicle)

        actor_list.append(sensor)

        sensor.listen(lambda data: process_img(data))

        time.sleep(30)

    finally:
        print('destroying actors')
        #sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for actor in actor_list:
            actor.destroy()
        print('done.')

if __name__ == '__main__':

    main()
