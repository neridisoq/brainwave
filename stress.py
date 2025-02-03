import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from matplotlib.widgets import Button
from datetime import datetime

class BrainwaveMonitor:
    def __init__(self, port='/dev/cu.usbmodem497D486D324D1', baudrate=115200):
        print(f"Initializing connection to {port}")
        self.serial_port = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        
        self.buffer_size = 100
        self.signal_quality = 0
        self.last_quality_print_time = time.time()
        self.quality_print_interval = 1.0  # Print every 1 second
        
        # Monitoring control
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.stress_values_during_monitoring = []
        
        # Initialize brainwave band buffers
        self.bands = {
            'Delta': deque(maxlen=self.buffer_size),     # 0.5 - 2.75 Hz
            'Theta': deque(maxlen=self.buffer_size),     # 3.5 - 6.75 Hz
            'Low-Alpha': deque(maxlen=self.buffer_size), # 7.5 - 9.25 Hz
            'High-Alpha': deque(maxlen=self.buffer_size),# 10 - 11.75 Hz
            'Low-Beta': deque(maxlen=self.buffer_size),  # 13 - 16.75 Hz
            'High-Beta': deque(maxlen=self.buffer_size), # 18 - 29.75 Hz
            'Low-Gamma': deque(maxlen=self.buffer_size), # 31 - 39.75 Hz
            'Mid-Gamma': deque(maxlen=self.buffer_size)  # 41 - 49.75 Hz
        }
        
        # Initialize metrics
        self.stress_index = deque(maxlen=self.buffer_size)
        self.packet_buffer = bytearray()
        self.packet_state = 'SYNC1'
        
        # Setup visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(6, 2, figure=self.fig)
        
        # Stress index plot
        self.stress_ax = self.fig.add_subplot(gs[0, :])
        self.stress_ax.set_title('Stress Index')
        self.stress_ax.set_ylim(0, 1000)
        self.stress_ax.grid(True, alpha=0.3)
        self.stress_line = self.stress_ax.plot([], [], 'r-', linewidth=2)[0]
        
        # Text displays with adjusted positions
        self.stress_text = self.stress_ax.text(
            0.75, 0.95, 'Stress Level: --',
            transform=self.stress_ax.transAxes,
            color='white',
            fontsize=12
        )
        
        self.quality_text = self.stress_ax.text(
            0.02, 0.95, 'Signal Quality: --',
            transform=self.stress_ax.transAxes,
            color='white',
            fontsize=12
        )
        
        self.avg_stress_text = self.stress_ax.text(
            0.02, 0.02, 'Average Stress: --',
            transform=self.stress_ax.transAxes,
            color='white',
            fontsize=12
        )
        
        # Create subplots for brainwave bands
        self.axes = {}
        for idx, (band_name, _) in enumerate(self.bands.items()):
            row = (idx // 2) + 1
            col = idx % 2
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_title(f'{band_name} Wave')
            ax.set_ylim(0, 1000000)
            ax.grid(True, alpha=0.3)
            self.axes[band_name] = {
                'ax': ax,
                'line': ax.plot([], [], label=band_name)[0]
            }
        
        # Control buttons with adjusted positions
        button_color = '#2ECC71'  # Green color
        self.start_button = plt.axes([0.35, 0.02, 0.1, 0.04])
        self.stop_button = plt.axes([0.55, 0.02, 0.1, 0.04])
        
        self.start_btn = Button(self.start_button, 'Start', color=button_color)
        self.stop_btn = Button(self.stop_button, 'Stop', color='#E74C3C')  # Red color
        self.stop_btn.set_active(False)  # Initially disabled
        
        self.start_btn.on_clicked(self.start_monitoring)
        self.stop_btn.on_clicked(self.stop_monitoring)
        
        plt.tight_layout()

    def print_signal_quality(self, quality):
        current_time = time.time()
        if current_time - self.last_quality_print_time >= self.quality_print_interval:
            quality_text = "Signal Quality: "
            if quality == 0:
                quality_text += "Excellent (0)"
            elif quality < 50:
                quality_text += f"Good ({quality})"
            elif quality < 100:
                quality_text += f"Fair ({quality})"
            elif quality < 200:
                quality_text += f"Poor ({quality})"
            else:
                quality_text += f"No Contact ({quality})"
            
            print(quality_text)
            self.last_quality_print_time = current_time

    def start_monitoring(self, event=None):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.stress_values_during_monitoring = []
            print("\n=== Monitoring Started ===")
            self.start_btn.set_active(False)
            self.stop_btn.set_active(True)
            
    def stop_monitoring(self, event=None):
        if self.is_monitoring:
            self.is_monitoring = False
            duration = (datetime.now() - self.monitoring_start_time).total_seconds()
            
            if self.stress_values_during_monitoring:
                avg_stress = sum(self.stress_values_during_monitoring) / len(self.stress_values_during_monitoring)
                result_text = f"\n=== Monitoring Results ===\n"
                result_text += f"Duration: {duration:.1f} seconds\n"
                result_text += f"Average Stress: {avg_stress:.1f}\n"
                result_text += f"Samples: {len(self.stress_values_during_monitoring)}\n"
                result_text += f"Min Stress: {min(self.stress_values_during_monitoring):.1f}\n"
                result_text += f"Max Stress: {max(self.stress_values_during_monitoring):.1f}\n"
                result_text += "======================="
                
                print(result_text)
                self.avg_stress_text.set_text(f'Avg Stress: {avg_stress:.1f}')
            
            self.start_btn.set_active(True)
            self.stop_btn.set_active(False)

    def calculate_metrics(self):
        if len(self.bands['Low-Alpha']) > 0 and len(self.bands['High-Alpha']) > 0 and \
           len(self.bands['Low-Beta']) > 0 and len(self.bands['High-Beta']) > 0:
            
            total_alpha = (self.bands['Low-Alpha'][-1] + self.bands['High-Alpha'][-1])
            total_beta = (self.bands['Low-Beta'][-1] + self.bands['High-Beta'][-1])
            
            if total_alpha > 0:
                beta_alpha_ratio = total_beta / total_alpha
                stress_value = min(1000, beta_alpha_ratio * 500)
                self.stress_index.append(stress_value)
                
                if self.is_monitoring:
                    self.stress_values_during_monitoring.append(stress_value)
            else:
                self.stress_index.append(0)
                if self.is_monitoring:
                    self.stress_values_during_monitoring.append(0)

    def parse_payload(self, payload):
        i = 0
        while i < len(payload):
            extended_code_level = 0
            while i < len(payload) and payload[i] == 0x55:
                extended_code_level += 1
                i += 1
            
            if i >= len(payload):
                break
            
            code = payload[i]
            i += 1
            
            if code & 0x80:
                if i >= len(payload):
                    break
                length = payload[i]
                i += 1
                value_data = payload[i:i+length]
                i += length
                
                if code == 0x83:  # ASIC_EEG_POWER
                    if len(value_data) >= 24:
                        for band_idx, (band_name, _) in enumerate(self.bands.items()):
                            start_idx = band_idx * 3
                            value = int.from_bytes(
                                value_data[start_idx:start_idx+3], 
                                byteorder='big',
                                signed=False
                            )
                            self.bands[band_name].append(value)
                        self.calculate_metrics()
            
            else:  # Single-byte value
                if i >= len(payload):
                    break
                value = payload[i]
                i += 1
                
                if code == 0x02:  # Poor signal quality
                    self.print_signal_quality(value)
                    
                    quality_text = "Signal Quality: "
                    if value == 0:
                        quality_text += "Excellent"
                        color = 'green'
                    elif value < 50:
                        quality_text += "Good"
                        color = 'green'
                    elif value < 100:
                        quality_text += "Fair"
                        color = 'yellow'
                    elif value < 200:
                        quality_text += "Poor"
                        color = 'red'
                    else:
                        quality_text += "No Contact"
                        color = 'red'
                    
                    self.quality_text.set_text(quality_text)
                    self.quality_text.set_color(color)

    def parse_packet(self):
        while len(self.packet_buffer) > 0:
            if self.packet_state == 'SYNC1':
                if self.packet_buffer[0] == 0xAA:
                    self.packet_state = 'SYNC2'
                self.packet_buffer = self.packet_buffer[1:]
            elif self.packet_state == 'SYNC2':
                if len(self.packet_buffer) >= 1:
                    if self.packet_buffer[0] == 0xAA:
                        self.packet_state = 'PLENGTH'
                    else:
                        self.packet_state = 'SYNC1'
                    self.packet_buffer = self.packet_buffer[1:]
            elif self.packet_state == 'PLENGTH':
                if len(self.packet_buffer) >= 1:
                    plength = self.packet_buffer[0]
                    if plength > 170:
                        self.packet_state = 'SYNC1'
                    else:
                        self.packet_state = 'PAYLOAD'
                    self.packet_buffer = self.packet_buffer[1:]
                    self.payload_length = plength
                    self.payload = bytearray()
            elif self.packet_state == 'PAYLOAD':
                if len(self.packet_buffer) >= self.payload_length:
                    self.payload.extend(self.packet_buffer[:self.payload_length])
                    self.packet_buffer = self.packet_buffer[self.payload_length:]
                    self.packet_state = 'CHECKSUM'
                else:
                    break
            elif self.packet_state == 'CHECKSUM':
                if len(self.packet_buffer) >= 1:
                    checksum = self.packet_buffer[0]
                    self.packet_buffer = self.packet_buffer[1:]
                    
                    calculated_checksum = (~sum(self.payload) & 0xFF)
                    if checksum == calculated_checksum:
                        self.parse_payload(self.payload)
                    
                    self.packet_state = 'SYNC1'
                else:
                    break

    def update_plot(self, frame):
        if self.serial_port.in_waiting:
            new_data = self.serial_port.read(self.serial_port.in_waiting)
            self.packet_buffer.extend(new_data)
            self.parse_packet()
            
        if len(self.stress_index) > 0:
            self.stress_line.set_data(range(len(self.stress_index)), 
                                    list(self.stress_index))
            self.stress_ax.set_xlim(0, len(self.stress_index))
            
            current_stress = self.stress_index[-1]
            stress_text = "Stress Level: "
            if current_stress < 300:
                stress_text += "Low"
                color = 'green'
            elif current_stress < 700:
                stress_text += "Medium"
                color = 'yellow'
            else:
                stress_text += "High"
                color = 'red'
            
            self.stress_text.set_text(stress_text)
            self.stress_text.set_color(color)
        
        for band_name, band_data in self.bands.items():
            if len(band_data) > 0:
                self.axes[band_name]['line'].set_data(
                    range(len(band_data)), 
                    list(band_data)
                )
                self.axes[band_name]['ax'].set_xlim(0, len(band_data))
        
        return ([self.stress_line] + 
                [data['line'] for data in self.axes.values()])

    def run(self):
        print("Starting brainwave monitoring system...")
        print("Press Start button to begin recording")
        ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=10,
            blit=True,
            cache_frame_data=False
        )
        plt.show()

    def close(self):
        if self.serial_port.is_open:
            self.serial_port.close()
        plt.close()

if __name__ == "__main__":
    try:
        monitor = BrainwaveMonitor()
        print("\n=== Brainwave Monitoring System ===")
        print("Signal Quality Guide:")
        print("  0        : Excellent - Perfect signal")
        print("  1-49     : Good - Clean signal")
        print("  50-99    : Fair - Slightly noisy")
        print("  100-199  : Poor - Very noisy")
        print("  200      : No Contact - Sensor not touching skin")
        print("\nPress Start button to begin recording...")
        
        monitor.run()
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        if 'monitor' in locals():
            monitor.close()
            print("Monitor closed successfully")