# farmer.py

# connector to the farms
from rlkit.envs.pyro_helper import pyro_connect

from rlkit.data_management.path_builder import PathBuilder

import threading as th
import time

# farmport = 20099

class farmlist:
    def __init__(self):
        self.list = []

    def generate(self):
        farmport = 20099
        def addressify(farmaddr,port):
            return farmaddr+':'+str(port)
        addresses = [addressify(farm[0],farmport) for farm in self.list]
        capacities = [farm[1] for farm in self.list]
        failures = [0 for i in range(len(capacities))]

        return addresses,capacities,failures

    def push(self, addr, capa):
        self.list.append((addr,capa))

fl = farmlist()


def reload_addr(farmlist_base={'nothing': 0}):
    global addresses,capacities,failures

    fl.list = []
    for item in farmlist_base:
        fl.push(item[0],item[1])

    addresses,capacities,failures = fl.generate()


class remoteEnv:
    def pretty(self,s):
        print(('(remoteEnv) {} ').format(self.id)+str(s))

    def __init__(self,fp,id): # fp = farm proxy
        self.fp = fp
        self.id = id
        self.last_observation = None
        self.current_path_builder = PathBuilder()

    def reset(self):
        return self.fp.reset(self.id)

    def step(self,actions):
        ret = self.fp.step(self.id, actions)
        if ret == False:
            self.pretty('env not found on farm side, might been released.')
            raise Exception('env not found on farm side, might been released.')
        return ret

    def set_spaces(self):
        from gym.spaces.box import Box
        import numpy as np
        a_s = self.fp.get_action_space(self.id)
        self.action_space = Box(np.array(a_s[0]),np.array(a_s[1]))
        o_s = self.fp.get_observation_space(self.id)
        self.observation_space = Box(np.array(o_s[0]),np.array(o_s[1]))

    def rel(self):
        count = 0
        while True: # releasing is important, so
            try:
                count+=1
                self.fp.rel(self.id)
                break
            except Exception as e:
                self.pretty('exception caught on rel()')
                self.pretty(e)
                time.sleep(3)
                if count>5:
                    self.pretty('failed to rel().')
                    break
                pass

        self.fp._pyroRelease()

    def get_last_observation(self):
        if self.last_observation:
            return self.last_observation
        else:
            return self.reset()
    
    def newPathBuilder(self):
        self.current_path_builder = PathBuilder()
    
    def __del__(self):
        self.rel()

class farmer:
    def reload_addr(self):
        self.pretty('reloading farm list...')
        reload_addr(self.farmlist_base)

    def pretty(self,s):
        print('(farmer) '+str(s))

    def __init__(self, farmlist_base):
        self.farmlist_base = farmlist_base
        self.reload_addr()
        for idx,address in enumerate(addresses):
            fp = pyro_connect(address,'farm')
            self.pretty('forced renewing... '+address)
            try:
                fp.forcerenew(capacities[idx])
                self.pretty('fp.forcerenew() success on '+address)
            except Exception as e:
                self.pretty('fp.forcerenew() failed on '+address)
                self.pretty(e)
                fp._pyroRelease()
                continue
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        ret = False

        import random # randomly sample to achieve load averaging
        # l = list(enumerate(addresses))
        l = list(range(len(addresses)))
        random.shuffle(l)

        for idx in l:
            time.sleep(0.1)
            address = addresses[idx]
            capacity = capacities[idx]

            if failures[idx]>0:
                # wait for a few more rounds upon failure,
                # to minimize overhead on querying busy instances
                failures[idx] -= 1
                continue
            else:
                fp = pyro_connect(address,'farm')
                try:
                    result = fp.acq(capacity)
                except Exception as e:
                    self.pretty('fp.acq() failed on '+address)
                    self.pretty(e)

                    fp._pyroRelease()
                    failures[idx] += 4
                    continue
                else:
                    if result == False: # no free ei
                        fp._pyroRelease() # destroy proxy
                        failures[idx] += 4
                        continue
                    else: # result is an id
                        eid = result
                        renv = remoteEnv(fp,eid) # build remoteEnv around the proxy
                        self.pretty('got one on {} id:{}'.format(address,eid))
                        ret = renv
                        break

        # ret is False if none of the farms has free ei
        return ret
