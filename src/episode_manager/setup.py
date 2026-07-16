from setuptools import find_packages, setup

package_name = 'episode_manager'

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
    description='Policy evaluation episode manager for CloudGripper V2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'episode_manager_node = episode_manager.episode_manager_node:main',
        ],
    },
)
