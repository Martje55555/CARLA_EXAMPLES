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
import logging
import argparse
import math

IM_WIDTH = 2560
IM_HEIGHT = 1440


def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3 / 255.0

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

def main():
    # list to store all actors to destroy together after simulation
    actor_list = []
    #vehicles_list = []
    walkers_list = []
    all_id = []

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
            cloudiness=50.0, 
			precipitation=100.0,
			precipitation_deposits=80.0,
			wind_intensity=100.0,
			sun_azimuth_angle=40.0,
			sun_altitude_angle=40.0,
			fog_density=10.0,
			fog_distance=50.0,
			wetness=50.0,
			fog_falloff=0.0,
			scattering_intensity=40.0, 
			mie_scattering_scale=10.0,
			rayleigh_scattering_scale=0.0331
        )

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

        # add pedestrians to the world
        world = client.get_world()
        blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprintsWalkers)

        spawn_points = []
        for i in range(1):
            spawn_point = carla.Transform()
            spawn_point.location = world.get_random_location_from_navigation()
            if (spawn_point.location != None):
                spawn_points.append(spawn_point)

        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        # apply the batch
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})


        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

        # apply the batch
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id


        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        world.wait_for_tick()

        #if magnitude(npc.get_location() - av.get_location()) < threshold:
        for i in range(0, len(all_actors), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # random max speed
            all_actors[i].set_max_speed(1 + random.random())

        # add more vehicles to the world
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 1):
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

        # set camera near hood of vehicle
        spawn_point = carla.Transform(carla.Location(x=.20, y=0, z=1.10))

        sensor = world.spawn_actor(vehicle_cam, spawn_point, attach_to=vehicle)

        actor_list.append(sensor)

        sensor.listen(lambda data: process_img(data))

        speed_factor = 1.0
        update_freq = 0.1 / speed_factor

        weather = Weather(world.get_weather())

        elapsed_time = 0.0

        timeout = time.time() + 30   # 60*5 = 5 minutes from now

        while True:
            if time.time() > timeout:
                break
            #test = test - 1
            #sys.stdout.write(str(elapsed_time))
            timestamp = world.wait_for_tick(seconds=30.0).timestamp
            elapsed_time += timestamp.delta_seconds
            if elapsed_time > update_freq:
                weather.tick(speed_factor * elapsed_time)
                world.set_weather(weather.weather)
                sys.stdout.write('\r' + str(weather) + 12 * ' ')
                sys.stdout.flush()
                elapsed_time = 0.0

        #time.sleep(30)
    finally:
        print('destroying actors')
        sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        # for actor in actor_list:
        #     actor.destroy()

        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
            client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        print('done.')

if __name__ == '__main__':

    main()
