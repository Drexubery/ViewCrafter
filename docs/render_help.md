## Point cloud render configurations
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--mode` | 'single_view_txt' | Currently we support 'single_view_txt' and 'single_view_specify' |
| `--traj_txt` | Required | Required for 'single_view_txt' mode, a txt file that specify camera trajectory |
| `--elevation` | 5. | The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment |
| `--center_scale` | 1. | Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image |
| `--d_theta` | 10. | Required for 'single_view_specify' mode, specify target theta angle as (theta + d_theta) |
| `--d_phi` | 30. | Required for 'single_view_specify' mode, specify target phi angle as (phi + d_phi) |
| `--d_r` | -.2 | Required for 'single_view_specify' mode, specify target radius as (r + r*dr) |

![fig](../assets/doc_world.png)