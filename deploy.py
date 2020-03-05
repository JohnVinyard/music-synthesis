from deploygraph import Requirement
from deploygraph.aws import Box, IngressRule, BaseSecurityGroup
import boto3

class TrainingBoxSecurityGroup(BaseSecurityGroup):
    def __init__(self):
        super().__init__(
            'training-box',
            'training boxes',
            boto3.client('ec2').describe_vpcs()['Vpcs'][0]['VpcId'],
            IngressRule(protocol='tcp', port=22, source='0.0.0.0/0'),
            IngressRule(protocol='tcp', port=8888, source='0.0.0.0/0'),
            IngressRule(protocol='tcp', port=443, source='0.0.0.0/0'))

class ConfiguredBox(Requirement):
    def __init__(self, box):
        super().__init__(box)
        self.box = box

    def data(self):
        return self.box.data()

    def fulfilled(self):
        connection = self.box.connection()
        try:
            with self._conda_env(connection):
                with connection.cd('music-synthesis'):
                    connection.run('python test.py')
        except Exception:
            return False

    def _conda_env(self, connection):
        return connection.prefix('source activate pytorch_p36')

    def fulfill(self):
        connection = self.box.connection()
        connection.run('sudo rm /var/lib/dpkg/lock-frontend', warn=True)
        connection.run('sudo rm /var/lib/dpkg/lock', warn=True)
        connection.run('sudo rm /var/cache/apt/archives/lock', warn=True)
        connection.run('sudo dpkg --configure -a')
        connection.sudo('apt-get install git')
        connection.run('rm -rf music-synthesis', warn=True)
        connection.run(
            'git clone https://github.com/JohnVinyard/music-synthesis.git', warn=True)

        with self._conda_env(connection):
            connection.run('conda remove llvmlite -y')
            connection.run('anaconda3/envs/pytorch_p36/bin/pip install zounds')
            connection.run('anaconda3/envs/pytorch_p36/bin/pip install lws')
            connection.run('conda install -c hcc -c conda-forge libsndfile=1.0.28 libsamplerate=0.1.8 libflac=1.3.1 libogg=1.3.2 librosa -y')


if __name__ == '__main__':
    group = TrainingBoxSecurityGroup()
    box = Box(
        instance_name='deep-learning',
        image_id='ami-025ed45832b817a35',
        instance_type='p2.xlarge',
        security_group=group,
        install_httping=False)
    configured = ConfiguredBox(box)
    configured()
    print(configured.data())
