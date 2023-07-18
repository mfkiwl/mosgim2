import os
import time
import numpy as np
import concurrent.futures
from datetime import datetime
from warnings import warn
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import h5py

from mosgim2.utils.geom_utils import sub_ionospheric

class Loader():
    
    def __init__(self, IPPh1, IPPh2):
        self.FIELDS = ['datetime', 'el', 'ipp_lat1', 'ipp_lon1', 'ipp_lat2', 'ipp_lon2', 'tec']
        self.DTYPE = (object, float, float, float, float, float, float)
        self.not_found_sites = []
        self.IPPh1 = IPPh1
        self.IPPh2 = IPPh2


class LoaderTxt(Loader):
    
    def __init__(self, root_dir, IPPh1, IPPh2):
        super().__init__(IPPh1, IPPh2)
        self.dformat = "%Y-%m-%dT%H:%M:%S"
        self.root_dir = root_dir


    def get_files(self, rootdir):
        """
        Root directroy must contain folders with site name 
        Inside subfolders are *.dat files for every satellite
        """
        result = defaultdict(list)
        for subdir, _, files in os.walk(rootdir):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".dat"):
                    site = filename[:4]
                    if site != subdir[-4:]:
                        raise ValueError(f'{site} in {subdir}. wrong site name')
                    result[site].append(filepath)
                else:
                    warn(f'{filepath} in {subdir} is not data file')
        for site in result:
            result[site].sort()
        return result


    def load_data(self, filepath):
        convert = lambda x: datetime.strptime(x.decode("utf-8"), self.dformat)
        data = np.genfromtxt(filepath, 
                             comments='#', 
                             names=self.FIELDS, 
                             dtype=self.DTYPE,
                             converters={"datetime": convert},  
                             )
        return data, filepath
    
    def __load_data_pool(self, filepath):
        return self.load_data(filepath), filepath

    def generate_data(self, sites=[]):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        for site, site_files in files.items():
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            count = 0
            st = time.time()
            for sat_file in site_files:
                try:
                    data, _ = self.load_data(sat_file)
                    count += 1
                    yield data, sat_file
                except Exception as e:
                    print(f'{sat_file} not processed. Reason: {e}')
            print(f'{site} contribute {count} files, takes {time.time() - st}')
            
    def generate_data_pool(self, sites=[], nworkers=1):
        files = self.get_files(self.root_dir)
        print(f'Collected {len(files)} sites')
        self.not_found_sites = sites[:]
        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            queue = []
            for site, site_files in files.items():
                if sites and not site in sites:
                    continue
                self.not_found_sites.remove(site)
                count = 0
                st = time.time()
                for sat_file in site_files:
                    try:
                        query = executor.submit(self.load_data, sat_file)
                        queue.append(query)
                    except Exception as e:
                        print(f'{sat_file} not processed. Reason: {e}')
                print(site)
            for v in concurrent.futures.as_completed(queue):
                yield v.result()


class LoaderHDF(Loader):
    
    def __init__(self, hdf_path, IPPh1, IPPh2):
        super().__init__(IPPh1, IPPh2)
        self.hdf_path = hdf_path
        
    def get_files(self):
        result = []
        for subdir, _, files in os.walk(self.hdf_path):
            for filename in files:
                filepath = Path(subdir) / filename
                if str(filepath).endswith(".h5"):
                    result.append(filepath)
        if len(result) != 1:
            msg = f'Must be exactly one hdf in {self.hdf_path} or subfolders'
            raise ValueError(msg)
        return result
    
    def __get_hdf_file(self):
        return self.get_files()[0]
    
    def generate_data(self, sites=[]):
        hdf_file = h5py.File(self.__get_hdf_file(), 'r')
        self.not_found_sites = sites[:]
        for site in hdf_file:
            if sites and not site in sites:
                continue
            self.not_found_sites.remove(site)
            slat = hdf_file[site].attrs['lat']
            slon = hdf_file[site].attrs['lon']
            st = time.time()
            count = 0
            for sat in hdf_file[site]:
                sat_data = hdf_file[site][sat]
                arr = np.empty((len(sat_data['tec']),), 
                               list(zip(self.FIELDS,self.DTYPE)))
                el = sat_data['elevation'][:]
                az = sat_data['azimuth'][:]
                ts = sat_data['timestamp'][:]
                ipp_lat1, ipp_lon1 = sub_ionospheric(slat, slon, self.IPPh1, az, el) 
                ipp_lat2, ipp_lon2 = sub_ionospheric(slat, slon, self.IPPh2, az, el)
           

                arr['datetime'] = np.array([datetime.utcfromtimestamp(float(t)) for t in ts])
                arr['el'] = np.rad2deg(el)
                arr['ipp_lat1'] = np.rad2deg(ipp_lat1)
                arr['ipp_lon1'] = np.rad2deg(ipp_lon1)

                arr['ipp_lat2'] = np.rad2deg(ipp_lat2)
                arr['ipp_lon2'] = np.rad2deg(ipp_lon2)

                arr['tec'] = sat_data['tec'][:]
                count += 1
                yield arr, sat + '_' + site
            print(f'{site} contribute {count} files, takes {time.time() - st}')

