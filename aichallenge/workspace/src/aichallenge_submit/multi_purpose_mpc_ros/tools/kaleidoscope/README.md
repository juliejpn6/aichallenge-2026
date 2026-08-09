# Kaleidoscope

[日本語](README.ja.md) | English

Kaleidoscope is the Python/Tk offline trajectory editor used by
`multi_purpose_mpc_ros`. Its implementation is kept in this directory so the
tool can later be published independently without moving MPC runtime code.

The supported environment is the organizer-provided Automotive AI Challenge
Docker image. Running directly in the host Python environment is not
supported. The editor is not a ROS node; it runs as a Python/Tk application
inside the container.

## Required repository layout

Use Kaleidoscope with the original AI Challenge repository layout unchanged.
Clone or copy this repository into
`aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/tools/kaleidoscope/`.
Renaming or moving `multi_purpose_mpc_ros`, its `env/` directory, the sibling
`aichallenge_submit_launch` package, or their map and trajectory files is not
supported. The user is responsible for preserving this layout.

## Run

On the host, change to the AI Challenge repository root and enter the Autoware
command container:

```bash
cd /path/to/aichallenge-2025
make autoware-bash
```

Inside the container, change to the Kaleidoscope directory and run it:

```bash
cd /aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/tools/kaleidoscope
python3 -m kaleidoscope
```

With the required repository layout in place, the command automatically opens
the built-in MPC trajectory and Lanelet2 map as a circular path. To select
files explicitly, run inside the container:

```bash
python3 -m kaleidoscope \
  --trajectory ../../env/final_ver3/traj_mincurv.csv \
  --osm ../../../aichallenge_submit_launch/map/lanelet2_map.osm \
  --circular
```

Do not create a host virtual environment, install the package on the host, or
run `python3 -m kaleidoscope` on the host.

## GUI troubleshooting

Run these commands on the host:

```bash
export XAUTHORITY=~/.Xauthority
./setup.bash doctor
```

If `DISPLAY` or `XAUTHORITY` reports a warning, follow the doctor output to fix
the X11 configuration, then enter the container again with
`make autoware-bash`.

## Runtime dependencies

The organizer Docker image must provide the following dependencies. Users do
not need to install them separately on the host.

- Python 3.10 or newer
- Tkinter (`python3-tk` on Ubuntu)
- defusedxml
- PyYAML

The editor GUI does not require `rclpy`, ROS topics, or ROS services. Before
publishing example maps or trajectories, document and verify their separate
redistribution terms.

## License

Kaleidoscope is licensed under the [Apache License 2.0](LICENSE).
