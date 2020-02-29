from deploygraph import Requirement
from deploygraph.aws import Box, PublicInternetSecurityGroup


class ConfiguredBox(Requirement):
    def __init__(self, box):
        super().__init__(box)
        self.box = box

    def data(self):
        pass

    def fulfilled(self):
        pass

    def fulfill(self):
        connection = self.box.connection()
        '''
        - install git
        - clone repo
        - activate conda environment
        - install requirements
        - run script
        '''
        connection.sudo('apt-get update --fix-missing')
        connection.sudo('apt-get install git')
        connection.run(
            'git clone https://github.com/JohnVinyard/music-synthesis.git')
        connection.run('conda activate ???')
        connection.run(
            'conda install libsndfile=1.0.28 libsamplerate=0.1.8 libflac=1.3.1 libogg=1.3.2')
        with connection.cd('music-synthesis'):
            connection.run('pip install requirements.txt')
            connection.run('python test.py')


if __name__ == '__main__':
    group = PublicInternetSecurityGroup()
    box = Box(
        instance_name='deep-learning',
        image_id='ami-025ed45832b817a35',
        instance_type='p2.xlarge',
        security_group=group,
        install_httping=False)
    box()
    print(box.data())
