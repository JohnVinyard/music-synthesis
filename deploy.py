from deploygraph import Requirement
from deploygraph.aws import Box, PublicInternetSecurityGroup


class ConfiguredBox(Requirement):
    def __init__(self, box):
        super().__init__(box)
        self.box = box

    def data(self):
        return self.box.data()

    def fulfilled(self):
        connection = self.box.connection
        try:
            with connection.cd('music-synthesis'):
                connection.run('python test.py')
        except Exception:
            return False

    def _conda_env(self):
        connection = self.box.connection()
        return connection.prefix('source activate pytorch_p36')

    def fulfill(self):
        connection = self.box.connection()
        '''
        - install git
        - clone repo
        - activate conda environment
        - install requirements
        - run script
        '''
        # connection.sudo('apt-get update --fix-missing')
        connection.run('sudo rm /var/lib/dpkg/lock-frontend', warn=True)
        connection.run('sudo rm /var/lib/dpkg/lock', warn=True)
        connection.run('sudo rm /var/cache/apt/archives/lock', warn=True)
        connection.run('sudo dpkg --configure -a')
        connection.sudo('apt-get install git')
        connection.run(
            'git clone https://github.com/JohnVinyard/music-synthesis.git', warn=True)
        with self._conda_env():
            connection.run(
                'conda install -c hcc -c conda-forge libsndfile=1.0.28 libsamplerate=0.1.8 libflac=1.3.1 libogg=1.3.2 -y')
            with connection.cd('music-synthesis'):
                connection.run('pip install --upgrade pip')
                connection.run('pip install -r requirements.txt')
                connection.run('python test.py')


if __name__ == '__main__':
    group = PublicInternetSecurityGroup()
    box = Box(
        instance_name='deep-learning',
        image_id='ami-025ed45832b817a35',
        instance_type='p2.xlarge',
        security_group=group,
        install_httping=False)
    configured = ConfiguredBox(box)
    configured()
    print(configured.data())
