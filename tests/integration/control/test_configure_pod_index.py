import pytest
import time

class TestConfigurePodIndex():
    def test_configure_pod_index(self, client, ready_pod_index):
        time.sleep(30) # Wait a little more, just in case.
        client.configure_index(ready_pod_index, replicas=1, pod_type='p1.x1')