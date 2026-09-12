import sys
from ctypes import windll
from time import sleep

from ursina import *
from ursina import application
import numpy as np
import os
from datetime import datetime
import csv
import cv2
from PIL import Image, ImageEnhance
import random

app = Ursina(title="Spiral Galaxy Generator - Training Dataset")
window.color = color.black
window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.collider_counter.enabled = False
window.size = [600, 600]
window.borderless = True

SHOW_INTERFACE = True
BATCH_CREATE = False

GALAXY_COLOR_PRESETS = [
    {
        'name': 'Classic Blue',
        'bulge_colors': [color.rgb(238, 237, 190), color.rgb(3, 136, 166), color.rgb(233, 242, 235)],
        'arm_colors': [color.rgb(97, 172, 248), color.rgb(249, 217, 114), color.rgb(221, 235, 225)],
        'dust_color': color.rgb(60, 40, 30),
        'bg_color': color.rgb(3, 10, 35),
    },
    {
        'name': 'Warm Yellow',
        'bulge_colors': [color.rgb(255, 240, 200), color.rgb(255, 200, 100), color.rgb(255, 220, 150)],
        'arm_colors': [color.rgb(255, 170, 133), color.rgb(225, 242, 0), color.rgb(255, 200, 100)],
        'dust_color': color.rgb(80, 50, 40),
        'bg_color': color.rgb(3, 2, 7),
    },
    {
        'name': 'Cold Purple',
        'bulge_colors': [color.rgb(200, 180, 255), color.rgb(100, 150, 255), color.rgb(150, 200, 255)],
        'arm_colors': [color.rgb(138, 43, 226), color.rgb(75, 0, 130), color.rgb(100, 149, 237)],
        'dust_color': color.rgb(50, 30, 60),
        'bg_color': color.rgb(18, 9, 38),
    },
    {
        'name': 'Red Giant',
        'bulge_colors': [color.rgb(255, 200, 200), color.rgb(255, 100, 100), color.rgb(255, 150, 150)],
        'arm_colors': [color.rgb(255, 28, 0), color.rgb(220, 60, 60), color.rgb(180, 40, 40)],
        'dust_color': color.rgb(70, 35, 35),
        'bg_color': color.rgb(28, 36, 39),
    },
    {
        'name': 'Green Mystery',
        'bulge_colors': [color.rgb(200, 255, 200), color.rgb(100, 255, 150), color.rgb(150, 255, 200)],
        'arm_colors': [color.rgb(0, 255, 127), color.rgb(50, 205, 50), color.rgb(144, 238, 144)],
        'dust_color': color.rgb(40, 60, 40),
        'bg_color': color.rgb(5, 15, 10),
    },
    {
        'name': 'Monochrome',
        'bulge_colors': [color.rgb(240, 240, 240), color.rgb(180, 180, 180), color.rgb(200, 200, 200)],
        'arm_colors': [color.rgb(220, 220, 220), color.rgb(160, 160, 160), color.rgb(140, 140, 140)],
        'dust_color': color.rgb(60, 60, 60),
        'bg_color': color.rgb(10 / 255, 10, 15),
    },
]


def calculate_pitch_angle(b):
    """
    Вычисляет угол закрутки (pitch angle) из параметра arms_curve
    Для логарифмической спирали: r = a * e^(b*theta)
    pitch_angle = arctan(b) в градусах
    """
    if b < 0.001:
        return 90.0
    pitch_rad = np.arctan(b)
    pitch_deg = np.degrees(pitch_rad)
    return pitch_deg


def calculate_b_from_pitch(pitch_degrees):
    """Обратная функция: b = tan(pitch)"""
    pitch_rad = np.radians(pitch_degrees)
    if pitch_rad < 0.001:
        return 100.0
    return np.tan(pitch_rad)


def add_noise(image, shot_noise=2.0, read_noise=1.5):
    """Добавляет шум: shot noise + read noise"""
    noise = np.random.normal(0, read_noise, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_psf(image, kernel_size=3):
    """Применяет PSF (размытие атмосферы/телескопа)"""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = cv2.filter2D(image[:, :, c].astype(np.float32), -1, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignetting(image, strength=0.35):
    """Добавляет виньетирование (затемнение к краям)"""
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    vignette = 1 - strength * (dist / max_dist) ** 2
    vignette = np.stack([vignette] * 3, axis=2)
    return np.clip(image * vignette, 0, 255).astype(np.uint8)


def add_sky_background(image, sky_level_range=(5, 25)):
    """Добавляет случайный фон неба"""
    sky_level = np.random.randint(*sky_level_range)
    return np.clip(image.astype(int) + sky_level, 0, 255).astype(np.uint8)


def post_process_screenshot(image):
    """Применяет все эффекты пост-обработки"""
    image = add_noise(image)
    image = apply_psf(image, kernel_size=3)
    image = apply_vignetting(image, strength=0.35)
    image = add_sky_background(image)
    return image


class GalaxyParams:
    def __init__(self):
        self.num_arms = int(random.uniform(1, 6))
        self.static_arm_curve = 0.12
        self.arms_curve = random.uniform(0.5, 0.7)
        self.tilt = 45
        self.has_bulge = round(random.uniform(0, 1))
        self.bulge_size = random.uniform(0.1, 0.3)
        self.bulge_sersic_n = 4.0
        self.star_count = 15000
        self.scale = 10
        self.camera_distance = 25
        self.max_theta = 7.0
        self.disk_scale_length = random.uniform(0.2, 0.3)
        self.dust_amount = random.uniform(0.0, 1.0)
        self.flocculence = random.uniform(0.0, 1.0)
        self.arm_width = 0.01
        self.asymmetry = 1
        self.color_preset = random.randint(0, len(GALAXY_COLOR_PRESETS) - 1)


params = GalaxyParams()

galaxy_entity = Entity()
dust_entity = Entity()

controls_text = Text(text='Galaxy Controls', position=(-0.85, 0.45), scale=2, color=color.azure)

slider_arms = Slider(min=1, max=8, default=params.num_arms, step=1, text='Arms', position=(-0.85, 0.38))
slider_static_curve = Slider(min=0.05, max=0.2, default=params.static_arm_curve, text='Scale Factor',
                             position=(-0.85, 0.31))
slider_arms_curve = Slider(min=0.09, max=0.7, default=params.arms_curve, text='Arms Curve (b)', position=(-0.85, 0.24))

pitch_text = Text(text='', position=(-0.85, 0.17), scale=1.5, color=color.cyan)

slider_bulge = Slider(min=0, max=1, step=1, default=params.has_bulge, text='Has Bulge', position=(-0.85, 0.10))
slider_bulge_size = Slider(min=0.1, max=0.3, default=params.bulge_size, text='Bulge Size', position=(-0.85, 0.03))
slider_tilt = Slider(min=0, max=90, default=round(random.uniform(0, 3)) * 30, text='Tilt (Deg)', position=(-0.85, -0.04))
slider_cam_dist = Slider(min=10, max=100, default=30, step=1, text='Camera Distance', position=(-0.85, -0.11))

slider_disk_scale = Slider(min=0.2, max=0.3, default=params.disk_scale_length, text='Disk Scale',
                           position=(-0.85, -0.18))
slider_dust = Slider(min=0, max=1, default=params.dust_amount, text='Dust Amount', position=(-0.85, -0.25))
slider_flocc = Slider(min=0, max=1, default=params.flocculence, text='Flocculence', position=(-0.85, -0.32))

slider_color_preset = Slider(min=0, max=len(GALAXY_COLOR_PRESETS) - 1, default=params.color_preset, step=1,
                             text='Color Preset', position=(-0.85, -0.39))
color_preset_text = Text(text='', position=(-0.85, -0.43), scale=1.2, color=color.yellow)

if not SHOW_INTERFACE:
    controls_text.enabled = False
    slider_arms.enabled = False
    slider_static_curve.enabled = False
    slider_arms_curve.enabled = False
    pitch_text.enabled = False
    slider_bulge.enabled = False
    slider_bulge_size.enabled = False
    slider_tilt.enabled = False
    slider_cam_dist.enabled = False
    slider_disk_scale.enabled = False
    slider_dust.enabled = False
    slider_flocc.enabled = False
    slider_color_preset.enabled = False
    color_preset_text.enabled = False

prev_values = {
    'arms': slider_arms.value, 'static': slider_static_curve.value,
    'curve': slider_arms_curve.value, 'bulge': slider_bulge.value,
    'bulge_size': slider_bulge_size.value, 'tilt': slider_tilt.value,
    'cam_dist': slider_cam_dist.value, 'disk_scale': slider_disk_scale.value,
    'dust': slider_dust.value, 'flocc': slider_flocc.value,
    'color_preset': slider_color_preset.value
}


def check_slider_changes():
    global prev_values
    current = {
        'arms': slider_arms.value, 'static': slider_static_curve.value,
        'curve': slider_arms_curve.value, 'bulge': slider_bulge.value,
        'bulge_size': slider_bulge_size.value, 'tilt': slider_tilt.value,
        'cam_dist': slider_cam_dist.value, 'disk_scale': slider_disk_scale.value,
        'dust': slider_dust.value, 'flocc': slider_flocc.value,
        'color_preset': slider_color_preset.value
    }
    changed = any(prev_values[k] != current[k] for k in prev_values)
    prev_values = current.copy()
    return changed


def save_metadata(filename_0, filename_angle):
    metadata_file = "screenshots/metadata.csv"
    file_exists = os.path.isfile(metadata_file)

    with open(metadata_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "id", "pitch_angle", "arms", "arms_curve", "static_arm_curve",
                "tilt", "has_bulge", "bulge_size", "disk_scale", "dust_amount",
                "flocculence", "color_preset", "filename_0", "filename_angle", "timestamp"
            ])

        pitch = calculate_pitch_angle(params.arms_curve)
        writer.writerow([
            datetime.now().strftime('%Y%m%d_%H%M%S'),
            f"{pitch:.2f}",
            params.num_arms,
            f"{params.arms_curve:.4f}",
            f"{params.static_arm_curve:.4f}",
            params.tilt,
            int(params.has_bulge),
            f"{params.bulge_size:.3f}",
            f"{params.disk_scale_length:.3f}",
            f"{params.dust_amount:.3f}",
            f"{params.flocculence:.3f}",
            params.color_preset,
            filename_0,
            filename_angle,
            datetime.now().isoformat()
        ])


def generate_galaxy():
    global galaxy_entity, dust_entity

    destroy(galaxy_entity)
    destroy(dust_entity)
    galaxy_entity = Entity()
    dust_entity = Entity()

    params.num_arms = int(slider_arms.value)
    params.static_arm_curve = slider_static_curve.value
    params.arms_curve = slider_arms_curve.value
    params.has_bulge = bool(int(slider_bulge.value))
    params.bulge_size = slider_bulge_size.value
    params.tilt = slider_tilt.value
    params.disk_scale_length = slider_disk_scale.value
    params.dust_amount = slider_dust.value
    params.flocculence = slider_flocc.value
    params.color_preset = int(slider_color_preset.value)

    current_preset = GALAXY_COLOR_PRESETS[params.color_preset]
    color_preset_text.text = f'Color: {current_preset["name"]}'

    bg = current_preset['bg_color']
    window.color = Vec4(
        bg[0] / 255.0,
        bg[1] / 255.0,
        bg[2] / 255.0,
        1.0
    )

    pitch_angle = calculate_pitch_angle(params.arms_curve)
    pitch_text.text = f'Pitch: {pitch_angle:.1f}° (b={params.arms_curve:.3f})'

    n_total = params.star_count
    x_all, y_all, z_all = [], [], []
    colors_all = []
    dust_x, dust_y, dust_z = [], [], []

    # ========== 1. ЯДРО (BULGE) ==========
    if params.has_bulge:
        n_bulge = int(n_total * 0.2)
        n_sersic = params.bulge_sersic_n
        r_eff = params.bulge_size

        for _ in range(n_bulge):
            angle_1 = np.random.uniform(0, 360)
            angle_2 = np.random.uniform(0, 360)

            u = np.random.random()
            r = r_eff * (-np.log(u)) ** (1 / n_sersic) * 0.5
            r = min(r, params.bulge_size)

            x = r * np.sin(np.radians(angle_1)) * np.cos(np.radians(angle_2))
            y = r * np.sin(np.radians(angle_1)) * np.sin(np.radians(angle_2))
            z = r * np.cos(np.radians(angle_1))

            z = np.clip(z, -0.05, 0.05)

            x_all.append(x)
            y_all.append(y)
            z_all.append(z)

            color_idx = random.randint(0, len(current_preset['bulge_colors']) - 1)
            colors_all.append(current_preset['bulge_colors'][color_idx])

    # ========== 2. СПИРАЛЬНЫЕ РУКАВА ==========
    theta_values = np.linspace(0, params.max_theta, 2000)
    n_arms_stars = n_total - (n_bulge if params.has_bulge else 0)
    stars_per_arm = n_arms_stars // params.num_arms
    angle_between_arms = 2 * np.pi / params.num_arms

    for arm_idx in range(params.num_arms):
        arm_offset = angle_between_arms * arm_idx
        arm_star_indices = np.random.choice(theta_values.size, stars_per_arm, replace=True)
        theta_arm = theta_values[arm_star_indices]

        radii = params.static_arm_curve * np.exp(params.arms_curve * theta_arm)

        brightness = np.exp(-radii / params.disk_scale_length)
        keep_mask = np.random.random(len(radii)) < brightness

        x_base = radii * np.cos(theta_arm)
        y_base = radii * np.sin(theta_arm)

        x_rot = x_base * np.cos(arm_offset) - y_base * np.sin(arm_offset)
        y_rot = x_base * np.sin(arm_offset) + y_base * np.cos(arm_offset)

        max_rand = 50
        cur_radius = np.random.uniform(15, 35)

        if params.flocculence > 0 and np.random.random() < params.flocculence:
            add_flocculent_segments(x_rot, y_rot, theta_arm, arm_offset,
                                    x_all, y_all, z_all, colors_all, current_preset)

        for i in range(len(theta_arm)):
            if not keep_mask[i]:
                continue

            visible = np.random.randint(0, max_rand)
            arm_brightness = 1.0 + params.asymmetry * np.sin(arm_idx * 1.7)

            if visible == 0 or (visible > max_rand / 5 and visible < max_rand):
                scatter_x = np.random.uniform(-cur_radius, cur_radius) / 500.0
                scatter_y = np.random.uniform(-cur_radius, cur_radius) / 500.0
                scatter_z = np.random.uniform(-4, 4) / 500.0

                x_all.append(x_rot[i] + scatter_x * arm_brightness)
                y_all.append(y_rot[i] + scatter_y * arm_brightness)
                z_all.append(scatter_z)

                color_idx = random.randint(0, len(current_preset['arm_colors']) - 1)
                colors_all.append(current_preset['arm_colors'][color_idx])

    x_all = np.array(x_all) if x_all else np.array([])
    y_all = np.array(y_all) if y_all else np.array([])
    z_all = np.array(z_all) if z_all else np.array([])

    if len(x_all) > 0:
        tilt_rad = np.radians(params.tilt)
        y_rot = y_all * np.cos(tilt_rad) - z_all * np.sin(tilt_rad)
        z_rot = y_all * np.sin(tilt_rad) + z_all * np.cos(tilt_rad)

        vertices = list(zip(x_all * params.scale, y_rot * params.scale, z_rot * params.scale))
        vertex_colors = []
        for c in colors_all:
            if isinstance(c, Color):
                vertex_colors.extend([float(c[0] / 255), float(c[1] / 255), float(c[2] / 255), 1.0])
            elif isinstance(c, (list, tuple)):
                if len(c) == 3:
                    vertex_colors.extend([float(c[0]), float(c[1]), float(c[2]), 1.0])
                elif len(c) == 4:
                    vertex_colors.extend([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
            elif isinstance(c, np.ndarray):
                if len(c) == 3:
                    vertex_colors.extend([float(c[0]), float(c[1]), float(c[2]), 1.0])
                elif len(c) == 4:
                    vertex_colors.extend([float(c[0]), float(c[1]), float(c[2]), float(c[3])])
            else:
                vertex_colors.extend([1.0, 1.0, 1.0, 1.0])

        vertex_colors = np.array(vertex_colors, dtype=np.float32)
        print(f"Вершин: {len(vertices)}, Цветов элементов: {len(vertex_colors)}")
        print(f"Соотношение: {len(vertex_colors) / len(vertices)} (должно быть 4.0)")

        vertex_colors = np.array(vertex_colors, dtype=np.float32)

        print(f"Вершин: {len(vertices)}, Цветов: {len(vertex_colors) // 3}")

        mesh = Mesh(
            vertices=vertices,
            colors=vertex_colors,
            mode='point',
            thickness=0.08
        )

        galaxy_entity.model = mesh

    if dust_x:
        dust_x = np.array(dust_x)
        dust_y = np.array(dust_y)
        dust_z = np.array(dust_z)
        tilt_rad = np.radians(params.tilt)
        dy_rot = dust_y * np.cos(tilt_rad) - dust_z * np.sin(tilt_rad)
        dz_rot = dust_y * np.sin(tilt_rad) + dust_z * np.cos(tilt_rad)

        dust_vertices = list(zip(dust_x * params.scale, dy_rot * params.scale, dz_rot * params.scale))
        dust_mesh = Mesh(vertices=dust_vertices, mode='point', thickness=0.12)
        dust_entity.model = dust_mesh
        dust_entity.color = current_preset['dust_color']


def add_flocculent_segments(x_base, y_base, theta, arm_offset, x_out, y_out, z_out, c_out, preset):
    """Добавляет короткие сегменты рукавов"""
    n_segments = np.random.randint(3, 8)

    for _ in range(n_segments):
        if np.random.random() > params.flocculence:
            continue
        start_idx = np.random.randint(0, len(theta) - 50)
        length = np.random.randint(20, 80)
        end_idx = min(start_idx + length, len(theta))

        for i in range(start_idx, end_idx, 3):
            if np.random.random() > 0.6:
                scatter = np.random.uniform(-0.02, 0.02)
                x_out.append(x_base[i] + scatter)
                y_out.append(y_base[i] + scatter)
                z_out.append(np.random.uniform(-0.01, 0.01))
                color_idx = random.randint(0, len(preset['arm_colors']) - 1)
                c_out.append(preset['arm_colors'][color_idx])


camera.position = (0, 20, -20)
camera.look_at((0, 0, 0))
camera_angle = 0
screenshot_text = Text(text='', position=(0, 0), origin=(0, 0), scale=2, color=color.green)
is_making_shot = False

def update():
    global camera_angle
    global is_making_shot

    if check_slider_changes():
        generate_galaxy()


    params.camera_distance = slider_cam_dist.value

    cam_dist = params.camera_distance
    cam_elevation = np.radians(params.tilt)

    x = cam_dist * np.sin(camera_angle) * np.cos(cam_elevation)
    y = cam_dist * np.sin(cam_elevation) + 5
    z = cam_dist * np.cos(camera_angle) * np.cos(cam_elevation)

    camera.position = (x, y, z)
    camera.look_at((0, 0, 0))

    if held_keys['p']:
        make_screenshot()

    if is_making_shot is False and BATCH_CREATE is True:
        is_making_shot = True
        invoke(make_screenshot, delay=1)


def make_screenshot():
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_tilt = params.tilt

    slider_tilt.value = 0
    generate_galaxy()

    invoke(take_screenshots_delayed, timestamp, original_tilt, delay=0.5)


def take_screenshots_delayed(timestamp, original_tilt):
    base.screenshot(namePrefix=f'screenshots/galaxy_{timestamp}_0.png', defaultFilename=0)

    slider_tilt.value = original_tilt
    generate_galaxy()

    invoke(take_screenshot_angle, timestamp, original_tilt, delay=0.5)


def take_screenshot_angle(timestamp, original_tilt):
    base.screenshot(namePrefix=f'screenshots/galaxy_{timestamp}_{int(original_tilt)}.png', defaultFilename=0)

    for fname in [f'screenshots/galaxy_{timestamp}_0.png',
                  f'screenshots/galaxy_{timestamp}_{int(original_tilt)}.png']:
        if os.path.exists(fname):
            img = cv2.imread(fname)
            img = post_process_screenshot(img)
            cv2.imwrite(fname, img)

    save_metadata(
        f'galaxy_{timestamp}_0.png',
        f'galaxy_{timestamp}_{int(original_tilt)}.png'
    )

    invoke(close_app, delay=0.5)


def close_app():
    application.quit()

generate_galaxy()
app.run()
