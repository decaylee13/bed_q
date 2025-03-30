import pygame
import sys
import random
from collections import deque
from typing import List, Tuple, Optional, Dict

# Import the existing classes (assuming they're in the same directory)
from patient import Patient
from bed import Bed


# Window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (144, 238, 144)
LIGHT_ORANGE = (255, 218, 185)
LIGHT_PINK = (255, 182, 193)
GRAY = (169, 169, 169)

# Bed & Layout constants
ROOM_WIDTH = 60
ROOM_HEIGHT = 40
ROOM_SPACING = 20

# Simulation parameters
PATIENT_ARRIVAL_INTERVAL = 0.6  # seconds
BASE_MOVE_SPEED = 4000.0        # pixels per second

class HospitalSimulation:
    """Simulation of hospital bed allocation using FIFO policy."""
    
    def __init__(self, num_beds: int = 24):
        # In this layout, with 4 wings × 6 beds each = 24 total
        self.num_beds = num_beds

        # Queues and references
        self.waiting_queue = deque()
        self.patients: Dict[int, Patient] = {}
        self.patient_positions: Dict[int, Tuple[float, float]] = {}
        self.patient_targets: Dict[int, Tuple[float, float]] = {}

        self.beds: List[Bed] = []
        self.bed_positions: List[Tuple[int, int]] = []  # (x,y) center for each bed

        # Timers, stats
        self.current_time = 0.0
        self.next_patient_arrival = 0.0
        self.next_patient_id = 1

        self.total_patients_treated = 0
        self.total_wait_time = 0.0
        
        # Build the hospital geometry
        self.setup_hospital_layout()

    def setup_hospital_layout(self):
        """
        Define geometry for wings, hallway, waiting area, etc.
        Then create the bed objects & positions aligned properly.
        """

        # --- Overall hallway: wide horizontal band across the middle ---
        self.hallway_height = 100
        # Center it vertically
        self.hallway_y = WINDOW_HEIGHT // 2 - self.hallway_height // 2

        # Entrance/exit area at bottom
        self.entrance_width = 220
        self.entrance_height = 80
        self.entrance_x = (WINDOW_WIDTH - self.entrance_width) // 2
        self.entrance_y = WINDOW_HEIGHT - (self.entrance_height + 20)

        # We'll define 4 wings (A, B, C, D) with consistent size.
        # Each wing will contain 2 columns x 3 rows of beds.
        # To give more space between columns, we first define a base column spacing:
        base_column_spacing = ROOM_SPACING
        # Then increase it to get more room
        self.WARD_COLUMN_SPACING = int(base_column_spacing * 2)  # originally twice ROOM_SPACING

        # Calculate the original wing block width (with a fixed margin of 40)
        original_wing_block_width = (2 * ROOM_WIDTH) + self.WARD_COLUMN_SPACING + 40
        # Now make the wing 50% longer horizontally:
        wing_block_width = int(original_wing_block_width * 1.5)
        # Wing block height remains the same:
        wing_block_height = (3 * ROOM_HEIGHT) + (2 * ROOM_SPACING) + 40

        self.wing_block_width = wing_block_width  # save for later use

        # Distance from hallway to the wings
        wing_vertical_offset = 50
        
        # Define each wing's position and color.
        self.wingA = {
            "name": "Wing A",
            "color": LIGHT_GREEN,
            "x": (WINDOW_WIDTH // 2) - wing_block_width - 120,
            "y": self.hallway_y - wing_block_height - wing_vertical_offset,
            "beds": 6
        }
        self.wingB = {
            "name": "Wing B",
            "color": LIGHT_BLUE,
            "x": (WINDOW_WIDTH // 2) + 120,
            "y": self.hallway_y - wing_block_height - wing_vertical_offset,
            "beds": 6
        }
        self.wingC = {
            "name": "Wing C",
            "color": LIGHT_ORANGE,
            "x": (WINDOW_WIDTH // 2) - wing_block_width - 120,
            "y": self.hallway_y + self.hallway_height + wing_vertical_offset,
            "beds": 6
        }
        self.wingD = {
            "name": "Wing D",
            "color": LIGHT_PINK,
            "x": (WINDOW_WIDTH // 2) + 120,
            "y": self.hallway_y + self.hallway_height + wing_vertical_offset,
            "beds": 6
        }
        
        self.wings = [self.wingA, self.wingB, self.wingC, self.wingD]

        # --- Waiting area: near the middle of the hallway ---
        self.waiting_area_width = 320
        self.waiting_area_height = 80
        self.waiting_area_x = (WINDOW_WIDTH - self.waiting_area_width) // 2
        # Center it vertically in the hallway
        self.waiting_area_y = self.hallway_y + (self.hallway_height - self.waiting_area_height) // 2
        self.waiting_area_color = (200, 200, 255)

        # Size for each "slot" in the waiting queue
        self.waiting_slot_width = 20
        self.waiting_slot_height = 20
        
        # Now create the actual bed objects and positions
        self.create_beds()

    def create_beds(self):
        """Create a total of 24 beds (6 in each wing) properly aligned."""
        bed_id = 0

        # For each wing, we want to position two columns such that:
        # - The left column is near the left edge (using a fixed left margin)
        # - The right column is near the right edge (using a fixed right margin)
        # This creates a wider gap between the two columns.
        fixed_left_margin = 20
        fixed_right_margin = 20

        for wing in self.wings:
            base_x = wing["x"]
            base_y = wing["y"]

            # The left column x position is:
            left_x = base_x + fixed_left_margin
            # The right column x position is computed by anchoring to the right side of the wing block:
            right_x = base_x + self.wing_block_width - fixed_right_margin - ROOM_WIDTH

            # Use a fixed top margin
            margin_top = 20
            start_y = base_y + margin_top

            # Place beds in 3 rows:
            for row in range(3):
                # Left column bed:
                bx_left = left_x
                by = start_y + row * (ROOM_HEIGHT + ROOM_SPACING)
                new_bed = Bed(bed_id)
                self.beds.append(new_bed)
                self.bed_positions.append((bx_left + ROOM_WIDTH // 2, by + ROOM_HEIGHT // 2))
                bed_id += 1

                # Right column bed:
                bx_right = right_x
                new_bed = Bed(bed_id)
                self.beds.append(new_bed)
                self.bed_positions.append((bx_right + ROOM_WIDTH // 2, by + ROOM_HEIGHT // 2))
                bed_id += 1

    def get_waiting_position(self, queue_index: int) -> Tuple[int, int]:
        """
        For a given position in the waiting queue, return the coordinate
        in the waiting area where that patient should stand.
        """
        max_per_row = self.waiting_area_width // self.waiting_slot_width
        row = queue_index // max_per_row
        col = queue_index % max_per_row

        margin_x = 10
        margin_y = 10
        x = self.waiting_area_x + margin_x + col * self.waiting_slot_width
        y = self.waiting_area_y + margin_y + row * self.waiting_slot_height

        x_max = self.waiting_area_x + self.waiting_area_width - self.waiting_slot_width
        y_max = self.waiting_area_y + self.waiting_area_height - self.waiting_slot_height
        x = min(x, x_max)
        y = min(y, y_max)
        return (x, y)

    def create_patient(self):
        """Create one new incoming patient from below the screen."""
        pid = self.next_patient_id
        self.next_patient_id += 1

        p = Patient(pid)
        p.status = "entering"
        self.patients[pid] = p

        start_x = WINDOW_WIDTH // 2
        start_y = WINDOW_HEIGHT + 30
        self.patient_positions[pid] = (start_x, start_y)

        entrance_target_x = WINDOW_WIDTH // 2
        entrance_target_y = self.entrance_y + self.entrance_height // 2
        self.patient_targets[pid] = (entrance_target_x, entrance_target_y)

        self.waiting_queue.append(pid)
        self.next_patient_arrival = self.current_time + PATIENT_ARRIVAL_INTERVAL

    def process_fifo_allocation(self):
        """Assign the first waiting patient to the first available bed, if any."""
        if not self.waiting_queue:
            return
        free_bed = next((b for b in self.beds if not b.is_occupied()), None)
        if free_bed is None:
            return

        patient_id = self.waiting_queue.popleft()
        patient = self.patients[patient_id]
        free_bed.assign_patient(patient)
        free_bed.calc_occupancy(patient)
        patient.status = "in bed"

        bx, by = self.bed_positions[free_bed.bed_id]
        self.patient_targets[patient_id] = (bx, by)

        self.total_wait_time += patient.wait_time
        self.update_waiting_positions()

    def process_discharges(self):
        """Any bed whose occupancy time is complete -> discharge patient."""
        for b in self.beds:
            if b.is_occupied() and b.time_occupied >= b.occupancy_delta:
                patient = b.current_patient
                patient.status = "discharged"
                self.patient_targets[patient.id] = (WINDOW_WIDTH // 2, WINDOW_HEIGHT + 60)
                b.discharge_patient()
                self.total_patients_treated += 1

    def update_waiting_positions(self):
        """After any queue changes, recalc targets for each waiting patient."""
        for i, pid in enumerate(self.waiting_queue):
            self.patient_targets[pid] = self.get_waiting_position(i)

    def update_bed_occupancy(self, dt: float):
        """Increment occupancy time for any occupied beds."""
        for b in self.beds:
            if b.is_occupied():
                b.time_occupied_increase(dt)

    def update_wait_times(self, dt: float):
        """Increase wait_time for each patient in the queue."""
        for pid in self.waiting_queue:
            if pid in self.patients:
                self.patients[pid].increase_wait_time(dt)

    def update_patient_positions(self, dt: float):
        """
        Move each patient closer to its target at a given speed.
        If a discharged patient moves off-screen, remove them from the sim.
        """
        to_remove = []
        for pid, patient in self.patients.items():
            if pid not in self.patient_positions or pid not in self.patient_targets:
                continue
            cx, cy = self.patient_positions[pid]
            tx, ty = self.patient_targets[pid]
            dx, dy = (tx - cx), (ty - cy)
            dist = (dx*dx + dy*dy) ** 0.5

            if dist < 1:
                if patient.status == "entering":
                    patient.status = "waiting"
                    self.update_waiting_positions()
                continue

            speed = BASE_MOVE_SPEED * dt
            if dist > 0:
                mvx = dx * speed / dist
                mvy = dy * speed / dist
                if abs(mvx) > abs(dx): mvx = dx
                if abs(mvy) > abs(dy): mvy = dy
                self.patient_positions[pid] = (cx + mvx, cy + mvy)

            if patient.status == "discharged" and cy > WINDOW_HEIGHT + 40:
                to_remove.append(pid)

        for pid in to_remove:
            del self.patients[pid]
            del self.patient_positions[pid]
            del self.patient_targets[pid]

    def update(self, dt: float):
        """Advance simulation time and process main logic."""
        self.current_time += dt
        if self.current_time >= self.next_patient_arrival:
            self.create_patient()
        self.update_bed_occupancy(dt)
        self.update_wait_times(dt)
        self.process_fifo_allocation()
        self.process_discharges()
        self.update_patient_positions(dt)

    @staticmethod
    def draw_bed(screen, bed_rect, bed_color, bed_id, bed_id_font, pillow_side="left"):
        """
        Draw a bed with rounded corners. The pillow and sheet line are drawn
        horizontally, but we flip them to the left or right side depending
        on `pillow_side`.
        """
        # 1) Main bed with rounded corners
        pygame.draw.rect(
            screen,
            bed_color,
            bed_rect,
            border_radius=8
        )

        # 2) Pillow
        pillow_width = bed_rect.width // 4
        pillow_height = bed_rect.height // 2
        pillow_y = bed_rect.y + (bed_rect.height - pillow_height) // 2  # center vertically

        if pillow_side == "left":
            # Pillow on the LEFT edge
            pillow_x = bed_rect.x + (bed_rect.width // 12)
            # Sheets line ~2/3 across from left
            sheet_line_x = bed_rect.x + (2 * bed_rect.width // 3)
        else:
            # Pillow on the RIGHT edge
            pillow_x = bed_rect.right - (bed_rect.width // 12) - pillow_width
            # Sheets line ~1/3 across from left (i.e. near the left side)
            sheet_line_x = bed_rect.x + (bed_rect.width // 3)

        # Draw the pillow
        pillow_rect = pygame.Rect(pillow_x, pillow_y, pillow_width, pillow_height)
        pygame.draw.rect(screen, WHITE, pillow_rect, border_radius=4)

        # 3) Vertical sheet line
        pygame.draw.line(
            screen,
            (200, 200, 200),
            (sheet_line_x, bed_rect.y),
            (sheet_line_x, bed_rect.bottom),
            width=2
        )

        # 4) Bed ID in the center
        id_str = bed_id_font.render(str(bed_id), True, BLACK)
        id_rect = id_str.get_rect(center=bed_rect.center)
        screen.blit(id_str, id_rect)

    def draw(self, screen):
        """Draw backgrounds, wings, beds, patients, and stats."""
        screen.fill(WHITE)
        font = pygame.font.Font(None, 28)

        pygame.draw.rect(
            screen, GRAY,
            (0, self.hallway_y, WINDOW_WIDTH, self.hallway_height)
        )
        
        corridor_x = WINDOW_WIDTH//2 - 40
        corridor_y = self.hallway_y + self.hallway_height
        corridor_w = 80
        corridor_h = self.entrance_y - corridor_y
        pygame.draw.rect(
            screen, GRAY,
            (corridor_x, corridor_y, corridor_w, corridor_h)
        )

        pygame.draw.rect(
            screen, (220,220,220),
            (self.entrance_x, self.entrance_y, self.entrance_width, self.entrance_height)
        )
        text = font.render("ENTRANCE/EXIT", True, BLACK)
        text_rect = text.get_rect(center=(
            self.entrance_x + self.entrance_width//2,
            self.entrance_y + self.entrance_height//2
        ))
        screen.blit(text, text_rect)

        h_text = font.render("HALLWAY", True, BLACK)
        h_rect = h_text.get_rect(center=(WINDOW_WIDTH//2, self.hallway_y + 20))
        screen.blit(h_text, h_rect)

        pygame.draw.rect(
            screen, self.waiting_area_color,
            (self.waiting_area_x, self.waiting_area_y,
             self.waiting_area_width, self.waiting_area_height)
        )
        small_font = pygame.font.Font(None, 20)
        wa_text = small_font.render("WAITING AREA", True, BLACK)
        wa_rect = wa_text.get_rect(center=(
            self.waiting_area_x + self.waiting_area_width//2,
            self.waiting_area_y + self.waiting_area_height//2
        ))
        screen.blit(wa_text, wa_rect)

        wing_label_font = pygame.font.Font(None, 30)
        bed_font = pygame.font.Font(None, 24)
        for wing in self.wings:
            wing_w = self.wing_block_width
            wing_h = (3 * ROOM_HEIGHT) + (2 * ROOM_SPACING) + 40
            pygame.draw.rect(
                screen, wing["color"],
                (wing["x"], wing["y"], wing_w, wing_h)
            )
            label_surf = wing_label_font.render(wing["name"], True, BLACK)
            screen.blit(label_surf, (wing["x"], wing["y"] - 30))

        for b, (cx, cy) in zip(self.beds, self.bed_positions):
            color = RED if b.is_occupied() else GREEN

            x_tl = cx - ROOM_WIDTH//2
            y_tl = cy - ROOM_HEIGHT//2
            bed_rect = pygame.Rect(x_tl, y_tl, ROOM_WIDTH, ROOM_HEIGHT)

            # Decide which side the pillow goes on based on bed_id
            # Even bed_id = left bed, Odd bed_id = right bed
            pillow_side = "left" if (b.bed_id % 2 == 0) else "right"

            # Draw the bed with a pillow on the chosen side
            HospitalSimulation.draw_bed(screen, bed_rect, color, b.bed_id, bed_font, pillow_side)

            id_surf = bed_font.render(str(b.bed_id), True, BLACK)
            id_rect = id_surf.get_rect(center=(cx, cy))
            screen.blit(id_surf, id_rect)

        for pid, (px, py) in self.patient_positions.items():
            patient = self.patients[pid]
            if patient.status == "waiting":
                color = BLUE
            elif patient.status == "in bed":
                color = (255,165,0)
            elif patient.status == "discharged":
                color = (128,128,128)
            else:
                color = (100,100,255)
            pygame.draw.circle(screen, color, (int(px), int(py)), 10)
            id_str = small_font.render(str(pid), True, WHITE)
            screen.blit(id_str, (px-5, py-5))

        stats = [
            f"Time: {self.current_time:.1f}s",
            f"Waiting: {len(self.waiting_queue)}",
            f"Treated: {self.total_patients_treated}",
            f"Avg Wait: {self.total_wait_time/max(1, self.total_patients_treated):.1f}s"
        ]
        y_offset = 10
        for line in stats:
            s_surf = font.render(line, True, BLACK)
            screen.blit(s_surf, (10, y_offset))
            y_offset += 30

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hospital Simulation - FIFO")
    clock = pygame.time.Clock()

    sim = HospitalSimulation(num_beds=24)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.update(dt)
        sim.draw(screen)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()