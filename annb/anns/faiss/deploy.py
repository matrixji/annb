from typing import Tuple

from annb.indexes import IndexUnderTestDeployment


class FaissIndexUnderTestDeployment(IndexUnderTestDeployment):
    def deploy(self, **kwargs) -> Tuple[str, str]:
        deployment_type = kwargs.get('deployment_type', 'builtin')
        if deployment_type == 'venv':
            kwargs['requirements'] = [
                'faiss-cpu', 'numpy'
            ]
        return super().deploy(**kwargs)


index_under_test_deployment = FaissIndexUnderTestDeployment
