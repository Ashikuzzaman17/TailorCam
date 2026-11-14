"""
Render every main TailorCam screen to PNG files without a camera, for
reviewing the design: reports/ui/<name>.png (1280 x 720, like the app).

  python scripts/ui_snapshots.py [out_folder]
"""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys
if ROOT not in _sys.path:
    _sys.path.insert(0, ROOT)
from paths import data_path  # noqa: E402  (dataset/, assets/, app_data/)
sys.path.insert(0, ROOT)
import output as O

W, H = 1280, 720
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'reports', 'ui')
os.makedirs(OUT, exist_ok=True)


def camera_frame():
    """A real photo letterboxed to the camera size, standing in for the webcam."""
    img = cv2.imread(data_path('real_photos', 'udb_4', 'front.jpg'))
    s = H / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * s), H))
    f = np.full((H, W, 3), 40, np.uint8)
    x0 = (W - img.shape[1]) // 2
    f[:, x0:x0 + img.shape[1]] = img
    return f


M = {'shoulder_width': 44.1, 'chest': 96.3, 'waist': 84.2, 'hip': 99.5, 'back_length': 45.0,
     'shirt_length': 73.0, 'sleeve_length': 60.2, 'neck': 38.4, 'thigh': 56.0}
RECORDS = [{'customer_name': n, 'date': '2026-10-20 14:3%d' % i, 'chest': 90 + i, 'waist': 78 + i}
           for i, n in enumerate(['Rahim Uddin', 'Karim Hasan', 'Nusrat Jahan', 'Farhana Akter'])]
CUSTOMERS = [{'id': i + 1, 'name': n, 'phone': '01711-00000%d' % i, 'gender': g, 'age': 30 + i,
              'weight_kg': 60 + 3 * i, 'height_cm': 165 + i}
             for i, (n, g) in enumerate([('Rahim Uddin', 'male'), ('Nusrat Jahan', 'female'),
                                         ('Karim Hasan', 'male'), ('Farhana Akter', 'female')])]
def body_mesh_image():
    """A real rendered 3D body (average SMPL shape), as the app shows after a scan."""
    try:
        import smplx
        m = smplx.SMPL(model_path=os.path.join(ROOT, 'model', 'smpl_weights', 'smpl_nochumpy', 'SMPL_NEUTRAL.pkl'))
        import torch
        pose = torch.zeros(1, 69); pose[0, 15 * 3 + 2], pose[0, 16 * 3 + 2] = -1.15, 1.15   # arms down
        return O.render_smpl_mesh(m(body_pose=pose).vertices[0].detach().numpy(), m.faces.astype(np.int64), 360, 540)
    except Exception as exc:
        print('mesh render skipped:', exc)
        return None


TAPE = {'neck': {'status': 'camera', 'expected_error': 1.3}, 'chest': {'status': 'tape', 'expected_error': 2.9},
        'waist': {'status': 'tape', 'expected_error': 3.6}, 'sleeve_length': {'status': 'tape', 'expected_error': 2.8}}


def shot(name, fn):
    try:
        out = fn()
        img = out[0] if isinstance(out, tuple) else out
        cv2.imwrite(os.path.join(OUT, f'{name}.png'), img)
        print('ok  ', name)
    except Exception as exc:
        print('FAIL', name, type(exc).__name__, exc)


if __name__ == '__main__':
    blank = lambda: np.zeros((H, W, 3), np.uint8)
    shot('1_splash', lambda: O.draw_splash_screen(blank(), 2.5))
    shot('2_home', lambda: O.draw_home_screen(blank(), RECORDS, t=3.0, stats={'customers': 42, 'scans': 118, 'orders': 17}))
    shot('3_customers', lambda: O.draw_customer_select_screen(blank(), CUSTOMERS))
    shot('4_details', lambda: O.draw_weight_age_overlay(camera_frame(), '68', '31', 'name', 'male', '', height_text='172', name_text='Rahim Udd'))
    shot('5_scan_guide', lambda: O.draw_photo_guide(camera_frame(), 'FRONT', 'Face the camera  --  arms slightly away from the body',
                                                    0.82, view_idx=0, height_est=172, total_views=2))
    shot('5b_countdown', lambda: O.draw_photo_guide(camera_frame(), 'FRONT', 'Face the camera  --  arms slightly away from the body',
                                                    0.91, countdown_val=2, view_idx=0, height_est=172, total_views=2))
    shot('5c_side_poor', lambda: O.draw_photo_guide(camera_frame(), 'LEFT', 'Turn to your left  --  arms relaxed by your sides',
                                                    0.30, view_idx=1, height_est=None, total_views=2, lighting_ok=False, guide_elapsed=9))
    shot('6_turn', lambda: O.draw_turn_instruction(camera_frame(), 'LEFT', 1, 2))
    shot('7_processing', lambda: O.draw_processing_screen(camera_frame()))
    MESH = body_mesh_image()
    shot('8_results', lambda: O.draw_results_overlay(blank(), M, mesh_img=MESH, reveal_elapsed=99, customer_name='Rahim Uddin', posture_score=0.86,
                                                     tape_status=TAPE, uncertainties={'chest': 1.2, 'waist': 2.8}))
    shot('8b_size_guide', lambda: O.draw_results_overlay(blank(), M, mesh_img=MESH, active_tab=1, customer_name='Rahim Uddin'))
    shot('8c_history', lambda: O.draw_results_overlay(blank(), M, mesh_img=MESH, active_tab=2, customer_name='Rahim Uddin',
                                                      history_records=[dict(M, scan_date='2026-10-1%d 12:00' % i) for i in range(3)]))
    shot('10_history_screen', lambda: O.draw_history_screen(blank(), [dict(M, name=c['name'], date='2026-10-1%d' % i) for i, c in enumerate(CUSTOMERS)]))
    shot('11_reports', lambda: O.draw_reports_screen(blank(), [dict(M, name=c['name'], date='2026-10-1%d' % i) for i, c in enumerate(CUSTOMERS)], 3.0))
    shot('12_settings', lambda: O.draw_settings_screen(blank()))
    shot('13_order', lambda: O.draw_order_screen(blank(), 'Rahim Uddin', M, 'shirt', 'slim fit', 'cotton', 'white', '1500', '2026-10-28',
                                                 'price', 'Karim', ['Karim', 'Salma']))
    shot('14_inventory', lambda: O.draw_inventory_screen(blank(), [{'id': 1, 'name': 'Cotton white', 'category': 'fabric', 'quantity': 12, 'unit': 'm', 'low_stock_threshold': 5},
                                                                   {'id': 2, 'name': 'Buttons black', 'category': 'accessory', 'quantity': 3, 'unit': 'pcs', 'low_stock_threshold': 10}], [2]))
    shot('15_height_input', lambda: O.draw_height_input_overlay(camera_frame(), '17'))
    shot('16_edit_value', lambda: O.draw_measurement_edit_overlay(blank(), 'chest', 'Chest Circ.', 96.3, '97'))
    import main as MA
    shot('17_feedback', lambda: MA._draw_feedback_overlay(blank(), M, MA._FEEDBACK_KEYS, 1, {'chest': '95', 'waist': '8'}))
    shot('9_toast', lambda: O.draw_toast(camera_frame(), 'Photo 1: whole body not visible (head to feet)', O.RED))
