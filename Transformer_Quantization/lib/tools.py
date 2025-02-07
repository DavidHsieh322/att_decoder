import time
import matplotlib.pyplot as plt
import numpy as np

def get_code_dtype(bits):
    if bits / 8 <= 1:
        dtype = np.uint8
    elif bits / 16 <= 1:
        dtype = np.uint16
    elif bits / 32 <= 1:
        dtype = np.uint32
    elif bits / 64 <= 1:
        dtype = np.uint64
    elif bits / 128 <= 1:
        dtype = np.uint128
    elif bits / 256 <= 1:
        dtype = np.uint256
    else:
        raise ValueError(f"Code range too big: use {bits} bits, but 256 bits max!")

    return dtype

class Timer():
    """Timer for each iteration"""
    def __init__(self, total_loops: int = None) -> None:
        """
        Inputs:
            total_loops: iteration number for calculating ETA
        """
        self.total_loops = total_loops
        self.times = 0
        
    def start(self) -> None:
        """Call it when start counting time"""
        self.start_time = time.perf_counter()
        self.times += 1
        
    def finish(self) -> None:
        """Call it when finish counting time"""
        self.elapsed_time = time.perf_counter() - self.start_time
        
        if self.total_loops != None:
            self.ETA = int(self.elapsed_time * (self.total_loops - self.times))
            secs = self.ETA %60
            mins = self.ETA //60 %60
            hours = self.ETA //60 //60 %24
            days = self.ETA //60 //60 //24
            self.ETA = f"{days}d {hours:02d}:{mins:02d}:{secs:02d}"
        
    def reset_times(self) -> None:
        """Call it when reset iteration number which has passed"""
        self.times = 0
        
class Plotter():
    """Plot curves according to the value in each iteration"""
    def __init__(self) -> None:
        """
        Inputs:
            None
        """
        self.data_list = []
        
    def append_data(self, data: list, fmt: str, label: str) -> None:
        """Append the y-axis values of a curve\n
        Inputs:
            data: a list contains the y-axis values of a curve
            fmt: curve color and format
            label: curve name
        """
        data.append(fmt)
        data.append(label)
        self.data_list.append(data)
        
    def reset_dataList(self) -> None:
        """Clear all curve data"""
        self.data_list = []
        
    def line_chart(
        self, 
        savepath: str, x_label: str, y_label: str, x_min: int, x_max: int, x_interval: int
    ) -> None:
        """Plot chart and save it\n
        Inputs:
            savepath: chart saving path
            x_label: x-axis title
            y_label: y-axis title
            x_min: x-axis minimum
            x_max: x-axis maximum
            x_interval: x-axis tick interval
        """
        plt.figure()
        
        for data in self.data_list:
            label = data.pop(-1)
            fmt = data.pop(-1)
            plt.plot(data, fmt, label=label)
            
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        xt_start = x_interval - (x_min % x_interval)
        xt_stop = x_max - x_min + 1
        xl_start = (x_min // x_interval + 1) * x_interval
        xl_stop = x_max + 1
        plt.xticks(
            [0] + [i for i in range(xt_start, xt_stop, x_interval)], 
            [x_min] + [j for j in range(xl_start, xl_stop, x_interval)]
        )
        plt.legend()
        plt.savefig(savepath)