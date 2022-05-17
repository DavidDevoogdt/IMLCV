import logging
import uuid
from time import sleep

import jobflow
from jobflow import Flow, JobStore, job, run_locally
from jobflow.managers.fireworks import JobFiretask, flow_to_workflow
from maggma.stores import MemoryStore, MongoStore

from fireworks import FWorker, LaunchPad
from fireworks.core.firework import Firework, Workflow
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.core.rocket_launcher import rapidfire as rpf
from fireworks.features.multi_launcher import launch_multiprocess
from fireworks.features.multi_launcher import rapidfire as rps1
from fireworks.queue.queue_launcher import launch_rocket_to_queue
from fireworks.queue.queue_launcher import rapidfire as rpf2


@job
def add(a, b):
    print(f"adding {a}+{b}")
    sleep(5)
    return a + b


unique_filename = str(uuid.uuid4())

store = JobStore(MongoStore(database=unique_filename,
                 collection_name='test'))

# store = JobStore(MemoryStore())

store.connect()

# task = JobFiretask(job=add, store=store)


flows = [Flow(add(1, i)) for i in range(5)]
fwjobs = [flow_to_workflow(
    flow=flow, store=store) for flow in flows]


lpad = LaunchPad()
lpad.reset("", require_password=False)

for fwj in fwjobs:
    lpad.add_wf(fwj)

# rpf(lpad)

# if True:
launch_multiprocess(launchpad=lpad, fworker=FWorker(),
                    nlaunches=0, num_jobs=6, sleep_time=1, loglvl="INFO", timeout=10000)


print('task end')


for item in list(store.query()):
    print(item)
