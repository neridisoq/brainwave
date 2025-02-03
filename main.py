import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

class MindSetMonitor:
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
        
        # Data buffers for different signals
        self.raw_data = deque(maxlen=512)  # Raw EEG data
        self.attention_data = deque(maxlen=100)  # Attention values
        self.meditation_data = deque(maxlen=100)  # Meditation values
        self.signal_quality = 0  # Poor signal quality value
        
        # Initialize visualization
        self.setup_visualization()
        
        # Packet parsing state
        self.packet_buffer = bytearray()
        self.packet_state = 'SYNC1'
        
    def setup_visualization(self):
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Raw EEG plot
        self.line_eeg, = self.ax1.plot([], [], 'g-', linewidth=1)
        self.ax1.set_ylim(-2048, 2048)
        self.ax1.set_title('Raw EEG')
        self.ax1.grid(True, alpha=0.3)
        
        # Attention plot
        self.line_attention, = self.ax2.plot([], [], 'b-', linewidth=1)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_title('Attention')
        self.ax2.grid(True, alpha=0.3)
        
        # Meditation plot
        self.line_meditation, = self.ax3.plot([], [], 'r-', linewidth=1)
        self.ax3.set_ylim(0, 100)
        self.ax3.set_title('Meditation')
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def parse_payload(self, payload):
        """Parse TGAM packet payload according to the protocol specification"""
        i = 0
        while i < len(payload):
            # Check for extended code level (0x55)
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
                
                if code == 0x80:  # Raw value
                    if len(value_data) >= 2:
                        raw_value = int.from_bytes(value_data[:2], byteorder='big', signed=True)
                        self.raw_data.append(raw_value)
                        print(f"Raw EEG: {raw_value}")
                
            else:  # Single-byte value
                value = payload[i] if i < len(payload) else 0
                i += 1
                
                if code == 0x02:  # Poor signal quality
                    self.signal_quality = value
                    print(f"Signal Quality: {value}")
                elif code == 0x04:  # Attention
                    self.attention_data.append(value)
                    print(f"Attention: {value}")
                elif code == 0x05:  # Meditation
                    self.meditation_data.append(value)
                    print(f"Meditation: {value}")

    def parse_packet(self):
        """Parse TGAM packets according to the protocol specification"""
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
                    if plength > 170:  # Maximum payload length check
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
                    
                    # Calculate checksum
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
            
        # Update raw EEG plot
        if len(self.raw_data) > 0:
            self.line_eeg.set_data(range(len(self.raw_data)), list(self.raw_data))
            self.ax1.set_xlim(0, len(self.raw_data))
            
        # Update attention plot
        if len(self.attention_data) > 0:
            self.line_attention.set_data(range(len(self.attention_data)), 
                                       list(self.attention_data))
            self.ax2.set_xlim(0, len(self.attention_data))
            
        # Update meditation plot
        if len(self.meditation_data) > 0:
            self.line_meditation.set_data(range(len(self.meditation_data)), 
                                        list(self.meditation_data))
            self.ax3.set_xlim(0, len(self.meditation_data))
            
        return self.line_eeg, self.line_attention, self.line_meditation

    def start_monitoring(self):
        print("Starting EEG monitoring...")
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
        monitor = MindSetMonitor()
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'monitor' in locals():
            monitor.close()