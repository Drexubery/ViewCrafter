## Point cloud render configurations
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--mode` | 'single_view_txt' | Currently we support 'single_view_txt' and 'single_view_target' mode|
| `--traj_txt` | None | Required for 'single_view_txt' mode, a txt file that specify camera trajectory |
| `--elevation` | 5. | The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment |
| `--center_scale` | 1. | Range: (0, 2]. Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image |
| `--d_theta` | 10. | Range: [-40, 40]. Required for 'single_view_target' mode, specify target theta angle as (theta + d_theta) |
| `--d_phi` | 30. | Range: [-45, 45]. Required for 'single_view_target' mode, specify target phi angle as (phi + d_phi) |
| `--d_r` | -.2 | Range: [-0.5, 0.5]. Required for 'single_view_target' mode, specify target radius as (r + r*dr) |
| `--d_x` | 0 | Range: [-200, 200]. Required for 'single_view_target' mode, '+' denotes pan right |
| `--d_y` | 0 | Range: [-100, 100]. Required for 'single_view_target' mode, '+' denotes pan up |
<hr>

![fig](../assets/doc_world.png)

The image above illustrates the definition of the world coordinate system.

**1.** Take a single reference image as an example, you first need to estimate an elevation angle `--elevation` that represents the angle at which the image was taken. A value greater than 0 indicates a top-down view, and it doesn't need to be precise.

**2.** The origin of the world coordinate system is by default defined at the point cloud corresponding to the center pixel of the reference image. You can adjust the position of the origin by modifying `--center_scale`; a value less than 1 brings the origin closer to the reference camera.

**3.** We use spherical coordinates to represent the camera pose. The initial camera is located at (r, 0, 0). You can specify a target camera pose by setting `--mode` as 'single_view_target'. As shown in the figure above, a positive `--d_phi` moves the camera to the right, a negative `--d_theta` moves the camera up, and a negative `--d_r` moves the camera forward (closer to the origin). You can also add panning motion by specifying `--d_x` and `--d_y`.  The program will interpolate a smooth trajectory between the initial pose and the target pose, then rendering the point cloud along that trajectory. Below shows some examples:
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td> --center_scale </td>
        <td> --d_phi </td>
        <td> --d_theta </td>
        <td> --d_r </td>
        <td>Render results</td>
    </tr>
   <tr>
  <td>
    0.5
  </td>
  <td>
    45.
  </td>
  <td>
    0.
  </td>
  <td>
    0.
  </td>
  <td>
    <img src=../assets/doc_tgt_scale5.gif width="250">
  </td>
  </tr>
   <tr>
  <td>
    1.
  </td>
  <td>
    45.
  </td>
  <td>
    0.
  </td>
  <td>
    0.
  </td>
  <td>
    <img src=../assets/doc_tgt_phi45.gif width="250">
  </td>
  </tr>
     <tr>
  <td>
    1.
  </td>
  <td>
    0.
  </td>
  <td>
    -30.
  </td>
  <td>
    0.
  </td>
  <td>
     <img src=../assets/doc_tgt_theta30.gif width="250">
  </td>
  </tr>
     <tr>
  <td>
    1.
  </td>
  <td>
    0.
  </td>
  <td>
    0.
  </td>
  <td>
   -0.5
  </td>
  <td>
    <img src=../assets/doc_tgt_r5.gif width="250">
  </td>
  </tr>
     <tr>
  <td>
    1.
  </td>
  <td>
    45.
  </td>
  <td>
    -30.
  </td>
  <td>
    -0.5
  </td>
  <td>
     <img src=../assets/doc_tgt_combine.gif width="250">
  </td>
  </tr>
</table>

**4.** You can also create a camera trajectory by specifying a sequence of d_phi, d_theta, d_r values. Set `--mode` as 'single_view_txt' and write the sequences in a txt file (example: [loop1.txt](../assets/loop1.txt)). The first line of the txt file should contain the target d_phi sequence, the second line the target d_theta sequence, and the third line the target d_r sequence. Each sequence should start with 0, and the length of each sequence should range from 2 to 25. Then, input the txt file path into `--traj_txt`. The program will interpolate a smooth trajectory based on the sequences you provide. Below shows some examples:
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td> Target sequences </td>
        <td> Trajectory visulization </td>
        <td>Render results</td>
    </tr>
   <tr>
  <td>
0 -3 -15 -20 -17 -5 0 <br>
0 -2 -5 -10 -8 -5 0 2 5 10 8 5 0 <br>
0  0
  </td>
  <td>
    <img src=../assets/loop1_traj.gif width="100">
  </td>
  <td>
    <img src=../assets/loop1_render.gif width="250">
  </td>
  </tr>
     <tr>
  <td>
0 3 10 20 17 10 0 <br>
0 -2 -8 -6 0 2 8 6 0 <br>
0 -0.02 -0.09 -0.18 -0.16 -0.09 0
  </td>
  
  <td>
    <img src=../assets/loop2_traj.gif width="100">
  </td>
  <td>
    <img src=../assets/loop2_render.gif width="250">
  </td>
  </tr>
         <tr>
  <td>
0  40 <br>
0 -1 -3 -7 -6 -4 0 1 3 7 6 4 0 -1 -3 -7 -6 -4 0 1 3 7 6 4 0 <br>
0  0
  </td>
  <td>
    <img src=../assets/wave_traj.gif width="100">
  </td>
  <td>
    <img src=../assets/wave_render.gif width="250">
  </td>
  </tr>
</table>

- **Tips:** A sequence in which the differences between adjacent values increase in one direction results in a smoother trajectory. Ensure that these differences are not too large; otherwise, they may lead to abrupt camera movements, causing the model to produce artifacts such as content drift.


