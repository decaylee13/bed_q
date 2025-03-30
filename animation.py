import pygame
import sys
import random
from collections import deque
from typing import List, Tuple, Optional, Dict

# Window size
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
FPS = 240

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BED_BLUE = (173, 216, 230)
LIGHT_BLUE = (91, 192, 235)
LIGHT_GREEN = (144, 238, 144)
LIGHT_ORANGE = (255, 218, 185)
LIGHT_PINK = (255, 182, 193)
GRAY = (169, 169, 169)
DARK_GREEN = (0, 128, 0)        # Darker shade of LIGHT_GREEN
DARK_ORANGE = (210, 105, 30)    # Darker shade of LIGHT_ORANGE
DARK_PINK = (199, 21, 133)      # Darker shade of LIGHT_PINK

# Bed & Layout constants
ROOM_WIDTH = 60
ROOM_HEIGHT = 40
ROOM_SPACING = 20

# Simulation parameters
PATIENT_ARRIVAL_INTERVAL = 0.6  # seconds
BASE_MOVE_SPEED = 600        # pixels per second
TOTAL_PATIENTS = 100

class Patient: 
    def __init__(self, id, severity=None):
        self.id = id
        self.wait_time = 0
        if severity is not None:
            self.severity = severity
        else:
            if random.random() < 0.7:
                self.severity = random.randint(1, 5)
            else:
                self.severity = random.randint(6, 10)
        self.status = "entering"
    
    def increase_wait_time(self, time_increment=1):
        self.wait_time += time_increment

    def get_features(self):
        return [self.wait_time, self.severity]

class Bed: 
    def __init__(self, bed_id, efficiency, allocation_strategy):
        self.bed_id = bed_id
        self.efficiency = efficiency
        self.time_occupied = 0 # how long has the patient been in bed? this changes; reset for new patients
        self.occupancy_delta = 0 # how long does the patient need to be in bed? this does not change; reset for new patients
        self.current_patient = None
        self.allocation_strategy = allocation_strategy
    
    def time_occupied_increase(self, time_increment = 1):
        self.time_occupied += time_increment

    def assign_patient(self, patient):
        """Assign a patient to this bed."""
        if self.current_patient:
            raise ValueError("Attempting to assign patient to an unavailable bed")
    
        self.current_patient = patient
        if self.allocation_strategy == 'FIFO Agent':
            self.occupancy_delta = 15*(patient.severity**2) / (self.efficiency)
        else:
            self.occupancy_delta = 5*(patient.severity**2) / (self.efficiency)
        
    def discharge_patient(self):
        """Discharge the current patient and make bed available."""
        self.reset()

    def is_occupied(self):
        return self.current_patient is not None

    def get_features(self):
        occupancy = self.is_occupied()
        return [occupancy, self.time_occupied, self.efficiency, self.occupancy_delta]

    def reset(self):
        """Reset the bed to initial state."""
        self.current_patient = None
        self.time_occupied = 0
        self.occupancy_delta = 5

class HospitalSimulation:
    """Simulation of hospital bed allocation using different strategies."""
    
    def __init__(self, num_beds: int = 24, patient_severities: Dict[int, int] = None, 
                 window_width=600, window_height=900, allocation_strategy='FIFO Agent'):
        self.num_beds = num_beds
        self.patient_severities = patient_severities or {}
        self.window_width = window_width
        self.window_height = window_height
        self.allocation_strategy = allocation_strategy

        # Queues and references
        self.waiting_queue = deque()
        self.patients: Dict[int, Patient] = {}
        self.patient_positions: Dict[int, Tuple[float, float]] = {}
        self.patient_targets: Dict[int, deque[Tuple[float, float]]] = {}

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
        """Define geometry for wings, hallway, waiting area, etc."""
        self.hallway_height = 140
        self.hallway_y = self.window_height // 2 - self.hallway_height // 2

        self.entrance_width = 220
        self.entrance_height = 80
        self.entrance_x = (self.window_width - self.entrance_width) // 2 
        self.entrance_y = self.window_height - (self.entrance_height + 20)

        base_column_spacing = ROOM_SPACING
        self.WARD_COLUMN_SPACING = int(base_column_spacing * 2)

        original_wing_block_width = (2 * ROOM_WIDTH) + self.WARD_COLUMN_SPACING + 40
        wing_block_width = int(original_wing_block_width * 1.2)
        wing_block_height = (3 * ROOM_HEIGHT) + (2 * ROOM_SPACING) + 40

        self.wing_block_width = wing_block_width

        wing_vertical_offset = 50
        
        self.wingA = {
            "name": "Wing A",
            "color": LIGHT_GREEN,
            "x": (self.window_width // 2) - wing_block_width - 60,
            "y": self.hallway_y - wing_block_height - wing_vertical_offset,
            "beds": 6
        }
        self.wingB = {
            "name": "Wing B",
            "color": LIGHT_BLUE,
            "x": (self.window_width // 2) + 60,
            "y": self.hallway_y - wing_block_height - wing_vertical_offset,
            "beds": 6
        }
        self.wingC = {
            "name": "Wing C",
            "color": LIGHT_ORANGE,
            "x": (self.window_width // 2) - wing_block_width - 60,
            "y": self.hallway_y + self.hallway_height + wing_vertical_offset,
            "beds": 6
        }
        self.wingD = {
            "name": "Wing D",
            "color": LIGHT_PINK,
            "x": (self.window_width // 2) + 60,
            "y": self.hallway_y + self.hallway_height + wing_vertical_offset,
            "beds": 6
        }
        
        self.wings = [self.wingA, self.wingB, self.wingC, self.wingD]

        self.waiting_area_width = 320
        self.waiting_area_height = 80
        self.waiting_area_x = (self.window_width - self.waiting_area_width) // 2
        self.waiting_area_y = self.hallway_y + (self.hallway_height - self.waiting_area_height) // 2
        self.waiting_area_color = (200, 200, 255)

        self.waiting_slot_width = 20
        self.waiting_slot_height = 20
        
        self.create_beds()

    def create_beds(self):
        """Create a total of 24 beds (6 in each wing) properly aligned."""
        bed_id = 0
        fixed_left_margin = 20
        fixed_right_margin = 20

        for wing in self.wings:
            base_x = wing["x"]
            base_y = wing["y"]
            left_x = base_x + fixed_left_margin
            right_x = base_x + self.wing_block_width - fixed_right_margin - ROOM_WIDTH
            margin_top = 20
            start_y = base_y + margin_top

            for row in range(3):
                bx_left = left_x
                by = start_y + row * (ROOM_HEIGHT + ROOM_SPACING)
                new_bed = Bed(bed_id, efficiency=(row+1)*5, allocation_strategy=self.allocation_strategy)
                self.beds.append(new_bed)
                self.bed_positions.append((bx_left + ROOM_WIDTH // 2, by + ROOM_HEIGHT // 2))
                bed_id += 1

                bx_right = right_x
                new_bed = Bed(bed_id, efficiency = (row+1)*5, allocation_strategy=self.allocation_strategy)
                self.beds.append(new_bed)
                self.bed_positions.append((bx_right + ROOM_WIDTH // 2, by + ROOM_HEIGHT // 2))
                bed_id += 1

    def get_waiting_position(self, queue_index: int) -> Tuple[int, int]:
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
        if self.next_patient_id > TOTAL_PATIENTS:
            return
        
        pid = self.next_patient_id
        self.next_patient_id += 1

        # Use predefined severity if available
        severity = self.patient_severities.get(pid, None)
        p = Patient(pid, severity)
        self.patients[pid] = p

        start_x = self.window_width // 2
        start_y = self.window_height + 30
        self.patient_positions[pid] = (start_x, start_y)

        entrance_target_x = self.window_width // 2
        entrance_target_y = self.entrance_y + self.entrance_height // 2
        hallway_center_x = self.window_width // 2
        hallway_center_y = self.hallway_y + self.hallway_height // 2

        self.patient_targets[pid] = deque()
        self.patient_targets[pid].append( (entrance_target_x, entrance_target_y) )
        self.patient_targets[pid].append( (hallway_center_x, hallway_center_y) )

        self.next_patient_arrival = self.current_time + PATIENT_ARRIVAL_INTERVAL

    def process_optimal_allocation(self):
        """Optimal allocation strategy assigning highest-severity patients to highest-efficiency beds."""
        # Get all available beds sorted by efficiency descending
        available_beds = [bed for bed in self.beds if not bed.is_occupied()]
        if not available_beds:
            return

        # Sort available beds by efficiency in descending order
        available_beds.sort(key=lambda x: x.efficiency, reverse=True)

        # Collect all patients in the waiting queue with their severity
        patients_in_queue = []
        for pid in self.waiting_queue:
            if pid in self.patients:
                patient = self.patients[pid]
                patients_in_queue.append((-patient.severity, pid))  # Negative for ascending sort later

        if not patients_in_queue:
            return

        # Sort patients by severity descending (using negative severity for ascending sort)
        patients_in_queue.sort()
        sorted_patient_ids = [pid for (sev, pid) in patients_in_queue]

        # Determine the number of possible assignments
        num_pairs = min(len(available_beds), len(sorted_patient_ids))
        if num_pairs == 0:
            return

        # Extract top patients and beds to assign
        patients_to_assign = sorted_patient_ids[:num_pairs]
        beds_to_assign = available_beds[:num_pairs]

        # Remove assigned patients from the waiting queue
        assigned_set = set(patients_to_assign)
        new_waiting_queue = deque([pid for pid in self.waiting_queue if pid not in assigned_set])
        self.waiting_queue = new_waiting_queue

        # Assign each patient to their respective bed
        for bed, pid in zip(beds_to_assign, patients_to_assign):
            if pid not in self.patients:
                continue  # Skip if patient was already removed (unlikely)
            patient = self.patients[pid]
            bed.assign_patient(patient)
            patient.status = "in bed"

            # Update patient movement waypoints
            bx, by = self.bed_positions[bed.bed_id]
            wing_index = bed.bed_id // 6
            wing = self.wings[wing_index]
            wing_corridor_x = wing['x'] + self.wing_block_width // 2
            hallway_center_y = self.hallway_y + self.hallway_height // 2

            waypoints = deque()
            waypoints.append((wing_corridor_x, hallway_center_y))
            waypoints.append((wing_corridor_x, by))
            waypoints.append((bx, by))

            self.patient_targets[pid] = waypoints

            self.total_wait_time += patient.wait_time

        # Update waiting positions for remaining patients
        self.update_waiting_positions()

    def process_fifo_allocation(self):
        if not self.waiting_queue:
            return
        free_bed = next((b for b in self.beds if not b.is_occupied()), None)
        if free_bed is None:
            return

        patient_id = self.waiting_queue.popleft()
        patient = self.patients[patient_id]
        free_bed.assign_patient(patient)
        patient.status = "in bed"

        bx, by = self.bed_positions[free_bed.bed_id]
        wing_index = free_bed.bed_id //6
        wing = self.wings[wing_index]
        wing_corridor_x = wing['x'] + self.wing_block_width //2
        hallway_center_y = self.hallway_y + self.hallway_height //2

        waypoints = deque()
        waypoints.append( (wing_corridor_x, hallway_center_y) )
        waypoints.append( (wing_corridor_x, by) )
        waypoints.append( (bx, by) )

        self.patient_targets[patient_id] = waypoints

        self.total_wait_time += patient.wait_time
        self.update_waiting_positions()

    def process_discharges(self):
        for b in self.beds:
            if b.is_occupied() and b.time_occupied >= b.occupancy_delta:
                patient = b.current_patient
                patient.status = "discharged"
                bx, by = self.bed_positions[b.bed_id]

                wing_index = b.bed_id //6
                wing = self.wings[wing_index]
                wing_corridor_x = wing['x'] + self.wing_block_width //2
                hallway_center_y = self.hallway_y + self.hallway_height //2
                exit_x = self.window_width //2
                exit_y = self.window_height +60

                waypoints = deque()
                waypoints.append( (wing_corridor_x, by) )
                waypoints.append( (wing_corridor_x, hallway_center_y) )
                waypoints.append( (exit_x, hallway_center_y) )
                waypoints.append( (exit_x, exit_y) )

                self.patient_targets[patient.id] = waypoints

                b.discharge_patient()
                self.total_patients_treated +=1

    def update_waiting_positions(self):
        for i, pid in enumerate(self.waiting_queue):
            if pid not in self.patients:
                continue
            target = self.get_waiting_position(i)
            if pid in self.patient_targets:
                self.patient_targets[pid].clear()
                self.patient_targets[pid].append(target)
            else:
                self.patient_targets[pid] = deque([target])

    def update_bed_occupancy(self, dt: float):
        for b in self.beds:
            if b.is_occupied():
                b.time_occupied_increase(dt)

    def update_wait_times(self, dt: float):
        for pid in self.waiting_queue:
            if pid in self.patients:
                self.patients[pid].increase_wait_time(dt)

    def update_patient_positions(self, dt: float):
        to_remove = []
        for pid, patient in self.patients.items():
            if pid not in self.patient_positions or pid not in self.patient_targets:
                continue

            cx, cy = self.patient_positions[pid]

            if not self.patient_targets[pid]:
                if patient.status == "entering":
                    patient.status = "waiting"
                    self.waiting_queue.append(pid)
                    self.update_waiting_positions()
                continue

            tx, ty = self.patient_targets[pid][0]
            dx = tx - cx
            dy = ty - cy
            dist = (dx**2 + dy**2)**0.5

            if dist < 1:
                self.patient_targets[pid].popleft()
                if not self.patient_targets[pid]:
                    if patient.status == "entering":
                        patient.status = "waiting"
                        self.waiting_queue.append(pid)
                        self.update_waiting_positions()
                    elif patient.status == "discharged":
                        pass
                continue

            speed = BASE_MOVE_SPEED * dt
            if speed > dist:
                self.patient_positions[pid] = (tx, ty)
            else:
                direction_x = dx / dist
                direction_y = dy / dist
                new_x = cx + direction_x * speed
                new_y = cy + direction_y * speed
                self.patient_positions[pid] = (new_x, new_y)

            if patient.status == "discharged" and cy > self.window_height + 40:
                to_remove.append(pid)

        for pid in to_remove:
            del self.patients[pid]
            del self.patient_positions[pid]
            del self.patient_targets[pid]

    def update(self, dt: float):
        self.current_time += dt
        if self.current_time >= self.next_patient_arrival:
            self.create_patient()
        self.update_bed_occupancy(dt)
        self.update_wait_times(dt)
        
        # Choose allocation strategy based on parameter
        if self.allocation_strategy == 'FIFO Agent':
            self.process_fifo_allocation()
        else:
            self.process_optimal_allocation()
            
        self.process_discharges()
        self.update_patient_positions(dt)

    @staticmethod
    def draw_bed(surface, bed_rect, bed_color, bed_id, bed_id_font, pillow_side="left"):
        pygame.draw.rect(surface, bed_color, bed_rect, border_radius=8)

        pillow_width = bed_rect.width // 4
        pillow_height = bed_rect.height // 2
        pillow_y = bed_rect.y + (bed_rect.height - pillow_height) // 2

        if pillow_side == "left":
            pillow_x = bed_rect.x + (bed_rect.width // 12)
            sheet_line_x = bed_rect.x + (2 * bed_rect.width // 3)
        else:
            pillow_x = bed_rect.right - (bed_rect.width // 12) - pillow_width
            sheet_line_x = bed_rect.x + (bed_rect.width // 3)

        pillow_rect = pygame.Rect(pillow_x, pillow_y, pillow_width, pillow_height)
        pygame.draw.rect(surface, WHITE, pillow_rect, border_radius=4)

        pygame.draw.line(surface, (200, 200, 200), (sheet_line_x, bed_rect.y), (sheet_line_x, bed_rect.bottom), width=2)

        id_str = bed_id_font.render(str(bed_id), True, BLACK)
        id_rect = id_str.get_rect(center=bed_rect.center)
        surface.blit(id_str, id_rect)

    def draw(self, surface):
        surface.fill((255, 248, 220))
        font = pygame.font.Font(None, 28)

        # Draw strategy title
        strategy_font = pygame.font.Font(None, 24)
        title = strategy_font.render(f"{self.allocation_strategy.upper()}", True, BLACK)
        title_rect = title.get_rect(center=(self.window_width//2, 20))
        surface.blit(title, title_rect)

        # Calculate the left and right edges of the wings
        left_edge = self.wingA["x"]  # Left edge of Wing A and C
        right_edge = self.wingB["x"] + self.wing_block_width  # Right edge of Wing B and D
        hallway_width = right_edge - left_edge

        pygame.draw.rect(surface, GRAY, (left_edge, self.hallway_y, hallway_width, self.hallway_height))

        corridor_w = 60
        corridor_x = self.window_width // 2 - corridor_w // 2
        corridor_y = self.hallway_y + self.hallway_height
        
        corridor_h = self.entrance_y - corridor_y
        pygame.draw.rect(surface, GRAY, (corridor_x, corridor_y, corridor_w, corridor_h))

        pygame.draw.rect(surface, (220,220,220), (self.entrance_x, self.entrance_y, self.entrance_width, self.entrance_height))
        text = font.render("ENTRANCE/EXIT", True, BLACK)
        text_rect = text.get_rect(center=(self.entrance_x + self.entrance_width//2, self.entrance_y + self.entrance_height//2))
        surface.blit(text, text_rect)

        h_text = font.render("HALLWAY", True, BLACK)
        h_rect = h_text.get_rect(center=(self.window_width//2, self.hallway_y - 20))
        surface.blit(h_text, h_rect)

        pygame.draw.rect(surface, self.waiting_area_color, (self.waiting_area_x, self.waiting_area_y, self.waiting_area_width, self.waiting_area_height))
        small_font = pygame.font.Font(None, 20)
        wa_text = small_font.render("WAITING AREA", True, BLACK)
        wa_rect = wa_text.get_rect(center=(self.waiting_area_x + self.waiting_area_width//2, self.waiting_area_y + self.waiting_area_height//2))
        surface.blit(wa_text, wa_rect)

        wing_label_font = pygame.font.Font(None, 30)
        bed_font = pygame.font.Font(None, 24)
        for wing in self.wings:
            wing_w = self.wing_block_width
            wing_h = (3 * ROOM_HEIGHT) + (2 * ROOM_SPACING) + 40
            pygame.draw.rect(surface, wing["color"], (wing["x"], wing["y"], wing_w, wing_h))
            
            # Choose text color based on wing name
            if wing["name"] == "Wing A":
                text_color = DARK_GREEN
            elif wing["name"] == "Wing B":
                text_color = BLUE
            elif wing["name"] == "Wing C":
                text_color = DARK_ORANGE
            elif wing["name"] == "Wing D":
                text_color = DARK_PINK
            else:
                text_color = BLACK
            
            label_surf = wing_label_font.render(wing["name"], True, text_color)
            
            # Center the text both horizontally and vertically in the wing rectangle
            label_x = wing["x"] + wing_w // 2 - label_surf.get_width() // 2
            label_y = wing["y"] + wing_h // 2 - label_surf.get_height() // 2
            surface.blit(label_surf, (label_x, label_y))
        for b, (cx, cy) in zip(self.beds, self.bed_positions):
            color = GREEN if b.is_occupied() else BED_BLUE

            x_tl = cx - ROOM_WIDTH//2
            y_tl = cy - ROOM_HEIGHT//2
            bed_rect = pygame.Rect(x_tl, y_tl, ROOM_WIDTH, ROOM_HEIGHT)
            pillow_side = "left" if (b.bed_id % 2 == 0) else "right"
            HospitalSimulation.draw_bed(surface, bed_rect, color, b.bed_id, bed_font, pillow_side)
            id_surf = bed_font.render(str(b.bed_id), True, BLACK)
            id_rect = id_surf.get_rect(center=(cx, cy))
            surface.blit(id_surf, id_rect)

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
            pygame.draw.circle(surface, color, (int(px), int(py)), 15)  # Increased radius to 15
            id_str = small_font.render(str(pid), True, WHITE)
            surface.blit(id_str, (px-8, py-8))  # Adjusted text position for the larger circle
        # Left column stats
        left_stats = [
            f"Time: {self.current_time:.1f}hrs",
            f"Waiting: {len(self.waiting_queue)}"
        ]
        # Right column stats
        right_stats = [
            f"Treated: {self.total_patients_treated}",
            f"Avg Wait: {self.total_wait_time/max(1, self.total_patients_treated):.1f}hrs"
        ]

        # Position for both columns
        x_left = 10
        x_right = 200
        y_offset = 40  # Start below the title

        # Draw left column
        for line in left_stats:
            s_surf = font.render(line, True, BLACK)
            surface.blit(s_surf, (x_left, y_offset))
            y_offset += 30

        # Reset y_offset for right column
        y_offset = 40  
        # Draw right column
        for line in right_stats:
            s_surf = font.render(line, True, BLACK)
            surface.blit(s_surf, (x_right, y_offset))
            y_offset += 30


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hospital Simulation - FIFO vs RL")
    clock = pygame.time.Clock()

    # Pre-generate patient severities for consistent comparison
    patient_severities = {}
    for pid in range(1, TOTAL_PATIENTS + 1):
        random.seed(pid)  # Ensure consistent severities across simulations
        if random.random() < 0.7:
            severity = random.randint(1, 5)
        else:
            severity = random.randint(6, 10)
        patient_severities[pid] = severity

    # Create two simulations with identical patient severities but different strategies
    sim_fifo = HospitalSimulation(
        num_beds=24,
        patient_severities=patient_severities,
        window_width=600,
        window_height=900,
        allocation_strategy='FIFO Agent'
    )
    sim_optimal = HospitalSimulation(
        num_beds=24,
        patient_severities=patient_severities,
        window_width=600,
        window_height=900,
        allocation_strategy='RL Agent'
    )

    # Surfaces for each simulation
    left_surface = pygame.Surface((600, 900))
    right_surface = pygame.Surface((600, 900))

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update both simulations
        sim_fifo.update(dt)
        sim_optimal.update(dt)

        # Draw to surfaces
        left_surface.fill((255, 248, 220))
        right_surface.fill((255, 248, 220))
        sim_fifo.draw(left_surface)
        sim_optimal.draw(right_surface)

        # Blit to main screen
        screen.blit(left_surface, (0, 0))
        screen.blit(right_surface, (600, 0))
        
        # Draw divider line
        pygame.draw.line(screen, BLACK, (600, 0), (600, 900), 2)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()