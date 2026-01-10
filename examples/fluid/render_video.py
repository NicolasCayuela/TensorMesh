import pyvista as pv
import os
import numpy as np
import glob
import subprocess

def create_video():
    print("Searching for VTU files...")
    files = sorted(glob.glob("vtk_output/frame_*.vtu"))
    if not files:
        print("No VTU files found in vtk_output/")
        return

    print(f"Found {len(files)} frames. Rendering...")
    
    # Create output directory for frames
    os.makedirs("frames", exist_ok=True)
    
    plotter = pv.Plotter(off_screen=True)
    
    # Setup camera and scalar bar once (optional, but good for consistency)
    # We need to load first mesh to set up range
    mesh = pv.read(files[0])
    
    for i, f in enumerate(files):
        mesh = pv.read(f)
        
        # Compute vorticity
        # Velocity is 2D in 3D array (z=0)
        # Check if velocity exists
        if "velocity" not in mesh.point_data:
            print(f"Frame {f} missing velocity data. Skipping.")
            continue
            
        # Ensure velocity is 3D for vorticity computation
        vel = mesh.point_data["velocity"]
        if vel.shape[1] == 2:
            # Pad with z=0
            vel_3d = np.zeros((vel.shape[0], 3), dtype=vel.dtype)
            vel_3d[:, :2] = vel
            mesh.point_data["velocity"] = vel_3d
            
        mesh = mesh.compute_derivative(scalars="velocity", gradient=False, vorticity=True)
        
        plotter.clear()
        # Plot vorticity Z component. 
        # Range: [-10, 10] to highlight structures.
        plotter.add_mesh(mesh, scalars="vorticity", cmap="RdBu", component=2, clim=[-10, 10], show_scalar_bar=True)
        plotter.view_xy()
        plotter.add_text(f"Step {i*5}", position='upper_left', font_size=10)
        
        frame_path = f"frames/frame_{i:04d}.png"
        plotter.screenshot(frame_path)
        if i % 10 == 0:
            print(f"Rendered frame {i}/{len(files)}")

    print("Encoding video with ffmpeg...")
    # generating mp4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "20",
        "-i", "frames/frame_%04d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "vortex_street.mp4"
    ]
    subprocess.run(cmd, check=True)
    print("Video generated: vortex_street.mp4")

if __name__ == "__main__":
    create_video()

