from setuptools import find_packages, setup

package_name = 'recorder'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CloudGripper',
    maintainer_email='todo@todo.com',
    description='LeRobot v3 dataset recorder node for CloudGripper V2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recorder_node = recorder.recorder_node:main',
        ],
    },
)
