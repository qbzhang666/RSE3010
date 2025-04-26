import tkinter as tk
from tkinter import ttk, messagebox

def calculate_strike_difference(a, b):
    a = a % 360
    b = b % 360
    diff = abs(a - b)
    return min(diff, 360 - diff)

def calculate_smr():
    try:
        # Get input values
        rmr_basic = float(rmr_entry.get())
        joint_strike = float(joint_strike_entry.get())
        joint_dip = float(joint_dip_entry.get())
        slope_strike = float(slope_strike_entry.get())
        slope_dip = float(slope_dip_entry.get())
        
        # Validate input ranges
        if not (0 <= rmr_basic <= 100):
            raise ValueError("RMR must be between 0 and 100")
        if not (0 <= joint_dip <= 90) or not (0 <= slope_dip <= 90):
            raise ValueError("Dip angles must be between 0 and 90 degrees")
        
        # Get adjustment factors
        excavation_method = excavation_var.get()
        failure_type = failure_var.get()

        # Calculate strike difference
        alpha = calculate_strike_difference(joint_strike, slope_strike)
        
        # Determine F1
        if alpha > 30:
            f1 = 1.0
        elif 20 < alpha <= 30:
            f1 = 0.85
        elif 10 < alpha <= 20:
            f1 = 0.70
        elif 5 < alpha <= 10:
            f1 = 0.50
        else:
            f1 = 0.30
            
        # Determine F2 (planar failure only)
        if joint_dip > 45:
            f2 = 1.0
        elif 30 < joint_dip <= 45:
            f2 = 0.85
        elif 20 < joint_dip <= 30:
            f2 = 0.70
        elif 10 < joint_dip <= 20:
            f2 = 0.50
        else:
            f2 = 0.30
            
        # Determine F3 (planar failure only)
        if joint_dip > slope_dip:
            f3 = -60
        elif joint_dip == slope_dip:
            f3 = -50
        else:
            f3 = 0
            
        # Determine F4
        excavation_f4 = {
            'natural': 0,
            'presplit': -5,
            'mechanical': -10,
            'poor blasting': -12
        }
        f4 = excavation_f4.get(excavation_method.lower(), 0)

        # Calculate SMR
        smr = rmr_basic + (f1 * f2 * f3) + f4
        
        # Update results
        results_text = (
            f"SMR Value: {smr:.2f}\n"
            f"Adjustment Factors:\n"
            f"F1: {f1}\nF2: {f2}\nF3: {f3}\nF4: {f4}"
        )
        result_label.config(text=results_text)
        
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))

# Create main window
root = tk.Tk()
root.title("Slope Mass Rating Calculator")
root.geometry("500x500")

# Create input frame
input_frame = ttk.LabelFrame(root, text="Input Parameters")
input_frame.pack(padx=10, pady=10, fill="both", expand=True)

# RMR Input
ttk.Label(input_frame, text="Basic RMR (0-100):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
rmr_entry = ttk.Entry(input_frame)
rmr_entry.grid(row=0, column=1, padx=5, pady=5)

# Joint Orientation
ttk.Label(input_frame, text="Joint Strike (°):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
joint_strike_entry = ttk.Entry(input_frame)
joint_strike_entry.grid(row=1, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Joint Dip (°):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
joint_dip_entry = ttk.Entry(input_frame)
joint_dip_entry.grid(row=2, column=1, padx=5, pady=5)

# Slope Orientation
ttk.Label(input_frame, text="Slope Strike (°):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
slope_strike_entry = ttk.Entry(input_frame)
slope_strike_entry.grid(row=3, column=1, padx=5, pady=5)

ttk.Label(input_frame, text="Slope Dip (°):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
slope_dip_entry = ttk.Entry(input_frame)
slope_dip_entry.grid(row=4, column=1, padx=5, pady=5)

# Excavation Method
ttk.Label(input_frame, text="Excavation Method:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
excavation_var = tk.StringVar()
excavation_combobox = ttk.Combobox(input_frame, textvariable=excavation_var, 
                                 values=["Natural", "Presplit", "Mechanical", "Poor Blasting"])
excavation_combobox.grid(row=5, column=1, padx=5, pady=5)
excavation_combobox.current(0)

# Failure Type
ttk.Label(input_frame, text="Failure Type:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
failure_var = tk.StringVar(value="Planar")
ttk.Combobox(input_frame, textvariable=failure_var, values=["Planar"], state="readonly").grid(row=6, column=1, padx=5, pady=5, sticky="w")

# Calculate Button
calculate_btn = ttk.Button(root, text="Calculate SMR", command=calculate_smr)
calculate_btn.pack(pady=10)

# Results Display
result_label = ttk.Label(root, text="SMR Value will be displayed here", 
                        relief="sunken", padding=10, wraplength=400)
result_label.pack(padx=10, pady=10, fill="both", expand=True)

# Set default values for testing
rmr_entry.insert(0, "50")
joint_strike_entry.insert(0, "85")
joint_dip_entry.insert(0, "50")
slope_strike_entry.insert(0, "90")
slope_dip_entry.insert(0, "60")

root.mainloop()
