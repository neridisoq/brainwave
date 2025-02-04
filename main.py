import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from matplotlib.widgets import Button
from datetime import datetime

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
        
        self.buffer_size = 512
        self.raw_data = deque(maxlen=self.buffer_size)
        self.attention_data = deque(maxlen=self.buffer_size)
        self.meditation_data = deque(maxlen=self.buffer_size)
        self.signal_quality = 0
        
        # Monitoring control
        self.is_monitoring = False
        self.monitoring_start_time = None
        self.attention_values = []
        self.meditation_values = []
        
        # Packet parsing state
        self.packet_buffer = bytearray()
        self.packet_state = 'SYNC1'
        
        # Setup visualization
        self.setup_visualization()
        
    def setup_visualization(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(4, 1)
        
        # Raw EEG plot
        self.ax1 = self.fig.add_subplot(gs[0])
        self.line_eeg, = self.ax1.plot([], [], 'g-', linewidth=1)
        self.ax1.set_ylim(-2048, 2048)
        self.ax1.set_title('Raw EEG')
        self.ax1.grid(True, alpha=0.3)
        
        # Attention plot
        self.ax2 = self.fig.add_subplot(gs[1])
        self.line_attention, = self.ax2.plot([], [], 'b-', linewidth=1)
        self.ax2.set_ylim(0, 100)
        self.ax2.set_title('Attention')
        self.ax2.grid(True, alpha=0.3)
        
        # Meditation plot
        self.ax3 = self.fig.add_subplot(gs[2])
        self.line_meditation, = self.ax3.plot([], [], 'r-', linewidth=1)
        self.ax3.set_ylim(0, 100)
        self.ax3.set_title('Meditation')
        self.ax3.grid(True, alpha=0.3)
        
        # Status information
        self.ax4 = self.fig.add_subplot(gs[3])
        self.ax4.axis('off')
        
        # Add buttons
        self.start_button = plt.axes([0.35, 0.02, 0.1, 0.04])
        self.stop_button = plt.axes([0.55, 0.02, 0.1, 0.04])
        
        self.btn_start = Button(self.start_button, 'Start', color='lightgreen')
        self.btn_stop = Button(self.stop_button, 'Stop', color='lightcoral')
        self.btn_stop.set_active(False)
        
        self.btn_start.on_clicked(self.start_recording)
        self.btn_stop.on_clicked(self.stop_recording)
        
        # Add text displays
        self.signal_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes)
        self.status_text = self.ax4.text(0.02, 0.7, '', transform=self.ax4.transAxes)
        self.results_text = self.ax4.text(0.02, 0.3, '', transform=self.ax4.transAxes)
        
        plt.tight_layout()

    def interpret_value(self, value):
        if value >= 80:
            return "Very High"
        elif value >= 60:
            return "High"
        elif value >= 40:
            return "Neutral"
        elif value >= 20:
            return "Low"
        else:
            return "Very Low"

    def start_recording(self, event):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()
            self.attention_values = []
            self.meditation_values = []
            self.btn_start.set_active(False)
            self.btn_stop.set_active(True)
            print("\n=== Recording Started ===")
            self.status_text.set_text("Status: Recording...")

    def stop_recording(self, event):
        if self.is_monitoring:
            self.is_monitoring = False
            duration = (datetime.now() - self.monitoring_start_time).total_seconds()
            
            if self.attention_values and self.meditation_values:
                avg_attention = sum(self.attention_values) / len(self.attention_values)
                avg_meditation = sum(self.meditation_values) / len(self.meditation_values)
                
                result_text = f"=== Recording Results ===\n"
                result_text += f"Duration: {duration:.1f} seconds\n"
                result_text += f"Average Attention: {avg_attention:.1f} ({self.interpret_value(avg_attention)})\n"
                result_text += f"Average Meditation: {avg_meditation:.1f} ({self.interpret_value(avg_meditation)})\n"
                result_text += f"Samples: {len(self.attention_values)}"
                
                print("\n" + result_text)
                self.results_text.set_text(result_text)
            
            self.btn_start.set_active(True)
            self.btn_stop.set_active(False)
            self.status_text.set_text("Status: Stopped")

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
                
            else:  # Single-byte value
                value = payload[i] if i < len(payload) else 0
                i += 1
                
                if code == 0x02:  # Poor signal quality
                    self.signal_quality = value
                    quality_text = f"Signal Quality: "
                    if value == 0:
                        quality_text += "Excellent"
                        color = 'green'
                    elif value < 50:
                        quality_text += "Good"
                        color = 'green'
                    elif value < 100:
                        quality_text += f"Fair ({value})"
                        color = 'yellow'
                    elif value < 200:
                        quality_text += f"Poor ({value})"
                        color = 'red'
                    else:
                        quality_text += "No Contact"
                        color = 'red'
                    
                    print(quality_text)
                    self.signal_text.set_text(quality_text)
                    self.signal_text.set_color(color)
                    
                elif code == 0x04:  # Attention
                    self.attention_data.append(value)
                    if self.is_monitoring:
                        self.attention_values.append(value)
                    
                elif code == 0x05:  # Meditation
                    self.meditation_data.append(value)
                    if self.is_monitoring:
                        self.meditation_values.append(value)

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
            
        if len(self.raw_data) > 0:
            self.line_eeg.set_data(range(len(self.raw_data)), list(self.raw_data))
            self.ax1.set_xlim(0, len(self.raw_data))
            
        if len(self.attention_data) > 0:
            self.line_attention.set_data(range(len(self.attention_data)), 
                                       list(self.attention_data))
            self.ax2.set_xlim(0, len(self.attention_data))
            
        if len(self.meditation_data) > 0:
            self.line_meditation.set_data(range(len(self.meditation_data)), 
                                        list(self.meditation_data))
            self.ax3.set_xlim(0, len(self.meditation_data))
            
        return self.line_eeg, self.line_attention, self.line_meditation

    def run(self):
        print("\n=== MindSet Monitor ===")
        print("Signal Quality Guide:")
        print("  0        : Excellent - Perfect signal")
        print("  1-49     : Good - Clean signal")
        print("  50-99    : Fair - Slightly noisy")
        print("  100-199  : Poor - Very noisy")
        print("  200      : No Contact - Sensor not touching skin")
        print("\nAttention/Meditation Level Guide:")
        print("  80-100   : Very High")
        print("  60-80    : High")
        print("  40-60    : Neutral")
        print("  20-40    : Low")
        print("  0-20     : Very Low")
        print("\nPress Start button to begin recording...")
        
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
        monitor = MindSetMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        if 'monitor' in locals():
            monitor.close()
            print("Monitor closed successfully")