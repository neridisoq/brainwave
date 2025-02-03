import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

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
        
        # Data buffers
        self.raw_data = deque(maxlen=512)
        self.signal_quality = 0
        self.buffer_size = 100
        
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
        
        # Packet parsing state
        self.packet_buffer = bytearray()
        self.packet_state = 'SYNC1'
        
        # Setup visualization
        self.setup_visualization()
    
    def setup_visualization(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 10))
        
        # Create grid of subplots for each brainwave band
        self.axes = {}
        grid_size = (4, 2)
        for idx, (band_name, _) in enumerate(self.bands.items()):
            ax = self.fig.add_subplot(grid_size[0], grid_size[1], idx + 1)
            ax.set_title(f'{band_name} Wave')
            ax.set_ylim(0, 1000000)  # Adjust based on your data range
            ax.grid(True, alpha=0.3)
            self.axes[band_name] = {
                'ax': ax,
                'line': ax.plot([], [], label=band_name)[0]
            }
        
        plt.tight_layout()
    
    def parse_payload(self, payload):
        i = 0
        while i < len(payload):
            # Handle extended code level
            extended_code_level = 0
            while i < len(payload) and payload[i] == 0x55:
                extended_code_level += 1
                i += 1
            
            if i >= len(payload):
                break
            
            code = payload[i]
            i += 1
            
            if code & 0x80:  # Multi-byte value
                if i >= len(payload):
                    break
                length = payload[i]
                i += 1
                value_data = payload[i:i+length]
                i += length
                
                if code == 0x83:  # ASIC_EEG_POWER
                    if len(value_data) >= 24:  # 8 bands × 3 bytes each
                        # Parse each frequency band (3 bytes each)
                        for band_idx, (band_name, _) in enumerate(self.bands.items()):
                            start_idx = band_idx * 3
                            value = int.from_bytes(
                                value_data[start_idx:start_idx+3], 
                                byteorder='big',
                                signed=False
                            )
                            self.bands[band_name].append(value)
                            print(f"{band_name}: {value}")
                
                elif code == 0x80:  # Raw value
                    if len(value_data) >= 2:
                        raw_value = int.from_bytes(value_data[:2], 
                                                 byteorder='big', 
                                                 signed=True)
                        self.raw_data.append(raw_value)
            
            else:  # Single-byte value
                if i >= len(payload):
                    break
                value = payload[i]
                i += 1
                
                if code == 0x02:  # Poor signal quality
                    self.signal_quality = value
                    if value > 0:
                        print(f"Poor signal quality: {value}")

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
        
        # Update each brainwave band plot
        for band_name, band_data in self.bands.items():
            if len(band_data) > 0:
                self.axes[band_name]['line'].set_data(
                    range(len(band_data)), 
                    list(band_data)
                )
                self.axes[band_name]['ax'].set_xlim(0, len(band_data))
        
        # Return all line objects for animation
        return [data['line'] for data in self.axes.values()]

    def start_monitoring(self):
        print("Starting brainwave monitoring...")
        ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=20,
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
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'monitor' in locals():
            monitor.close()