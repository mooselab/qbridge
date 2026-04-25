from abc import abstractmethod
import qiskit.providers.fake_provider
import os.path, pkgutil,importlib,inspect
from qiskit import Aer, IBMQ


class abstract_interface:

    @abstractmethod
    def get_result(self, backend, input_data, number_of_runs, seed):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def get_collective_result(self, backend, input_data, number_of_runs, seed, iterations):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def get_happy_scene(self):
        raise NotImplementedError("Please Implement this method")






class BackendFactory:

    def __init__(self):

        pkgpath = os.path.dirname(qiskit.providers.fake_provider.backends.__file__)

        backends = {}
        for bk in [name for _, name, _ in pkgutil.iter_modules([pkgpath])]:
            md = importlib.import_module("qiskit.providers.fake_provider.backends." + bk)
            clsmembers = inspect.getmembers(md, inspect.isclass)
            for name in [x[0] for x in clsmembers]:
                backends[name] = md

        self.backends = backends

    def get_backends_list(self):
        return list(self.backends.keys())

    def initialize_backend(self,name=None,real=None):
        if real:
            if name:
                IBMQ.save_account('69f16f44de2970998b5b3a8dc5dc8622760f117523728ab8201ad4f07013b2ae8829944e812b7f453bc0ff943ae345b52ed356255a30c686ab3f52a8de2ca894', overwrite=True)
                provider = IBMQ.load_account()
                device = provider.get_backend(name)
                return device
            else:
                print("Name Required")

        else:
            if name==None:
                return Aer.get_backend('aer_simulator')
            else:
                al = self.backends[name]
                class_ = getattr(al, name)
                return class_()




