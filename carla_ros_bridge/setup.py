"""
Setup for carla_ros_bridge
"""
import os
from glob import glob
ROS_VERSION = int(os.environ['ROS_VERSION'])

if ROS_VERSION == 1:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(packages=['carla_ros_bridge'], package_dir={'': 'src'})

    setup(**d)

elif ROS_VERSION == 2:
    from setuptools import setup

    package_name = 'carla_ros_bridge'
    setup(
        name=package_name,
        version='0.0.0',
        packages=[package_name],
        data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                    ('share/' + package_name, ['package.xml']),
                    (os.path.join('share', package_name), glob('launch/*.launch.py')),
                    (os.path.join('share', package_name + '/test'), glob('test/test_objects.json'))],
        install_requires=['setuptools'],
        zip_safe=True,
        maintainer='CARLA Simulator Team',
        maintainer_email='carla.simulator@gmail.com',
        description='CARLA ROS2 bridge',
        license='MIT',
        tests_require=['pytest'],
        entry_points={
            'console_scripts': [
                'bridge = carla_ros_bridge.bridge:main',
                'spawn_ground_truth = carla_ros_bridge.spawn_ground_truth:main',
                'spawn_rgb_lidar = carla_ros_bridge.spawn_rgb_lidar:main',
                'lap_monitor = carla_ros_bridge.lap_monitor:main',
                'record_evaluation = carla_ros_bridge.record_evaluation:main',
                'evaluate_run = carla_ros_bridge.evaluate_run:main',
            ],
        },
        package_dir={'': 'src'},
        package_data={'': ['CARLA_VERSION']},
        include_package_data=True
    )
