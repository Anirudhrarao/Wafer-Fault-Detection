from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
       Desc: This method will read all pypi from requirements remove newline '\n'
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='Wafer Fault Detection',
    version='2.0.0',
    author='Anirudhra',
    author_email='raorudhra16@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirement.txt')
)