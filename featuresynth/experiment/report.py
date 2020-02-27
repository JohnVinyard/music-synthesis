from deploygraph import AlwaysUpdateException, Requirement
from deploygraph.aws import S3Bucket, CorsConfig
from ..util import device
from botocore.config import Config
from botocore import UNSIGNED
import boto3
from io import BytesIO
import json
import os
import torch
from itertools import islice
import numpy as np
from matplotlib import pyplot as plt
from uuid import uuid4
import pprint
import datetime


class StaticResource(Requirement):
    def __init__(self, bucket, flo, filename, content_type):
        super().__init__(bucket)
        self.filename = filename
        self.flo = flo
        self.content_type = content_type
        self.client = boto3.client('s3')
        self.bucket = bucket

        config = Config()
        config.signature_version = UNSIGNED
        self.url_generator = boto3.client('s3', config=config)

    def fulfill(self):
        self.client.put_object(
            Bucket=self.bucket.bucket_name,
            Body=self.flo,
            Key=self.filename,
            ACL='public-read',
            ContentType=self.content_type
        )

    def fulfilled(self):
        raise AlwaysUpdateException()

    def data(self):
        uri = self.url_generator.generate_presigned_url(
            'get_object',
            ExpiresIn=0,
            Params={'Bucket': self.bucket.bucket_name, 'Key': self.filename})
        return {'uri': uri}


class StaticApp(Requirement):
    def __init__(self, bucket, cors, *static_resources):
        super().__init__(bucket, cors, *static_resources)
        self.static_resources = static_resources
        self.bucket = bucket

    def fulfill(self):
        pass

    def fulfilled(self):
        return True

    def data(self):
        bucket_data = self.bucket.data()
        return {
            'uris': [sr.data()['uri'] for sr in self.static_resources],
            'bucket_endpoint': bucket_data['endpoint'],
            'bucket_name': bucket_data['bucket']
        }


def generate_image(data):
    fig = plt.figure()
    if data.ndim == 1:
        plt.plot(data)
    elif data.ndim == 2:
        data = np.asarray(data).real
        mat = plt.matshow(np.rot90(data), cmap=plt.cm.viridis)
        mat.axes.get_xaxis().set_visible(False)
        mat.axes.get_yaxis().set_visible(False)
    else:
        raise ValueError('cannot handle dimensions > 2')
    bio = BytesIO()
    plt.savefig(bio, bbox_inches='tight', pad_inches=0, format='png')
    bio.seek(0)
    fig.clf()
    plt.close('all')
    return bio


class Report(object):
    def __init__(self, experiment):
        self.experiment = experiment

    @property
    def bucket_name(self):
        base = self.experiment.__class__.__name__.lower()
        return f'generation-report-{base}'

    def _local_file(self, filename):
        path, _ = os.path.split(__file__)
        filepath = os.path.join(path, filename)
        with open(filepath, 'rb') as f:
            return BytesIO(f.read())

    def generate(
            self, data_source, anchor_feature, n_examples, sr, regenerate=True):
        bucket = S3Bucket(self.bucket_name, 'us-west-1')
        cors = CorsConfig(bucket)

        app_data = {
            'title': self.experiment.__class__.__name__,
            'comments': self.experiment.__doc__,
            'date': datetime.datetime.utcnow().isoformat(),
            'results': []
        }

        dynamic_static_resources = []

        if regenerate:
            # delete the bucket and start all over
            client = boto3.resource('s3')
            try:
                b = client.Bucket(self.bucket_name)
                b.objects.all().delete()
                b.delete()
                print(f'deleted bucket {self.bucket_name}')
            except boto3.client('s3').exceptions.NoSuchBucket:
                print(f'{self.bucket_name} does not yet exist')

            self.experiment = self.experiment.to(device)
            self.experiment.resume()

            batch_stream = islice(
                self.experiment.batch_stream(1, data_source, anchor_feature),
                n_examples)

            for batch in batch_stream:
                base_name = uuid4().hex[:6]

                samples, features = self.experiment.preprocess_batch(batch)

                real_spec = features[0].T
                real_repr = self.experiment.from_audio(samples, sr)
                features = torch.from_numpy(features).to(device)
                fake = self.experiment.generator(features).data.cpu().numpy()
                audio_repr = self.experiment.audio_representation(fake, sr)
                real_audio = real_repr.listen()
                fake_audio = audio_repr.listen()

                resources = {
                    'feature': StaticResource(
                        bucket,
                        generate_image(real_spec),
                        f'{base_name}_feature.png',
                        'image/png'
                    ),
                    'real_spectrogram': StaticResource(
                        bucket,
                        generate_image(real_repr.display()),
                        f'{base_name}_real_spec.png',
                        'image/png'
                    ),
                    'fake_spectrogram': StaticResource(
                        bucket,
                        generate_image(audio_repr.display()),
                        f'{base_name}_fake_spec.png',
                        'image/png'
                    ),
                    'real_audio': StaticResource(
                        bucket,
                        real_audio.encode(fmt='OGG', subtype='VORBIS'),
                        f'{base_name}_real_audio.ogg',
                        'audio/ogg'
                    ),
                    'fake_audio': StaticResource(
                        bucket,
                        fake_audio.encode(fmt='OGG', subtype='VORBIS'),
                        f'{base_name}_fake_audio.ogg',
                        'audio/ogg'
                    )
                }
                dynamic_static_resources.extend(resources.values())
                app_data['results'].append(
                    {k: v.filename for k, v in resources.items()})
                print(base_name)

            report_data = StaticResource(
                bucket,
                BytesIO(json.dumps(app_data).encode()),
                'report_data.json',
                'application/json')
            dynamic_static_resources.append(report_data)

        html_source = self._local_file('report_template.html')
        js_source = self._local_file('app.js')

        html = StaticResource(
            bucket, html_source, 'index.html', 'text/html')
        js = StaticResource(
            bucket,
            js_source,
            'app.js',
            'application/javascript')

        report = StaticApp(
            bucket, cors, html, js, *dynamic_static_resources)
        report()
        pprint.pprint(report.data())
