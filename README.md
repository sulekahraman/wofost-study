# wofost-study
## Setup
Requires python3.7 or later.

Create a virtual environment and activate it.
```
pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

Install dependencies listed in requirements.txt.
```
pip install -r requirements.txt
```

## Modifying the `pcse` package 
We observed that the weather data values might be missing for some dates and locations, for which WOFOST throws an error. Therefore we updated the `pcse` source code to interpolate the missing weather values by using the values from the previous days.
To use this imputation method for weather, you can simply change `venv/lib/python3.7/site-packages/pcse/base/weather.py` with the `weather.py` in this repo. 


## Virtual machine setup
To view jupyter notebooks running on a remote machine on your local browser,
you can use port forwarding.

For convenience, first create aliases for these commands on your `~/.bashrc`.

On the remote machine:
```
alias jp="jupyter notebook --no-browser"
```

On your local machine:
```
alias port_forward="ssh -N -f -L localhost:XXXX:localhost:YYYY user@remote"
alias show_ports="netstat -vanp --tcp"
```
Source your `.bashrc`s on both machines. Now can view the ports in use by the command `show_ports`. 
Choose any available `XXXX` port on local machine and `YYYY` port on the remote machine.


After aliasing, run the following commands to start the jupyter notebook at port=YYYY on the remote machine,
and to forwards ports such that you can view the notebook on your browser at localhost:XXXX

On the remote machine:
```
jp --port=XXXX
```

On the local machine:
```
port_forward
```

You will need to repeat the above two lines every time you disconnect from the remote machine.
To avoid killing the notebook when ssh connection is broken, you can use tmux.

