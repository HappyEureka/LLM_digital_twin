class SimTime():
    def __init__(
            self, 
            interval=30, # minutes
            open_hour=10, 
            open_minute=0,
            close_hour=20,
            close_minute=0,):
        self.hour = open_hour
        self.minute = open_minute
        self.open_hour = open_hour
        self.open_minute = open_minute
        self.close_hour = close_hour
        self.close_minute = close_minute
        self.interval = interval
        # potentially add weekday
    
    def get_open_time(self):
        return f"{self.open_hour}:{self.open_minute:02d}"
    
    def get_close_time(self):
        return f"{self.close_hour}:{self.close_minute:02d}"

    def add(self, hour, minute):
        self.hour += hour
        self.minute += minute
        self.hour += self.minute // 60
        self.minute = self.minute % 60

    def get_time(self):
        return (self.hour, self.minute)

    def get_time_float(self):
        return self.hour + self.minute / 60
    
    def get_time_string(self):
        return f"{self.hour}:{self.minute:02d}"
    
    def step(self):
        self.add(0, self.interval)

    def pass_close(self):
        return self.hour >= self.close_hour or (self.hour == self.close_hour and self.minute >= self.close_minute)