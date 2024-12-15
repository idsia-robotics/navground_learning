from setuptools import setup, find_namespace_packages
import glob

package_name = 'navground_learning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_namespace_packages(where='.', include=['navground.learning']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jerome Guzzi',
    maintainer_email='jerome@idsia.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
)
