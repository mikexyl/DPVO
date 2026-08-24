from glob import glob
import os

from setuptools import find_packages, setup


package_name = "dpvo_multi_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mikexyl",
    maintainer_email="mikexyl@users.noreply.github.com",
    description="Distributed classic loop closure transport for DPVO.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dpvo_multi_robot_node = dpvo_multi_robot.bootstrap:main",
            "dpvo_centralized_pgo = dpvo_multi_robot.bootstrap:centralized_pgo_main",
            "dpvo_euroc_player = dpvo_multi_robot.bootstrap:euroc_player_main",
        ],
    },
)
