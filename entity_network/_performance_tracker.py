from time import time

import pandas as pd

# TODO: implement logger

class operation_tracker():

    def __init__(self):

        self.process_time = pd.DataFrame(columns=['caller','file','function','description','duration_seconds'])

    def reset_time(self):

        self.timer_start = time()


    def track(self, caller, file, function, description):

        # calculate time in seconds since last invocation
        duration = time()-self.timer_start

        # print to standard out for users to track long running processes
        print(f'caller={caller}, file={file}, function={function}, description={description}, duration_seconds={duration}')

        # record all processes
        df = pd.DataFrame([[caller, file, function, description, duration]], columns=self.process_time.columns)
        self.process_time = pd.concat([self.process_time, df], ignore_index=True)
        self.process_time = self.process_time.sort_values(by='duration_seconds', ascending=False)

        # reset timer for next invocation of tracking
        self.timer_start = time()