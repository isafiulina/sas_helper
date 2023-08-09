import os
import re
import pytraj as pt
import nglview as nv
import urllib
import matplotlib
import ipywidgets as widgets
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from time import sleep
from ipywidgets import interact, widgets, fixed, VBox
import time
import subprocess
import glob

def get_created_files(command):
    initial_files = os.listdir()
    initial_time = time.time()
    subprocess.run(command, shell=True)
    final_files = os.listdir()
    created_files = []
    for file in final_files:
        if file not in initial_files:
            created_files.append(file)
        else:
            file_path = os.path.join(os.getcwd(), file)
            file_creation_time = os.path.getctime(file_path)
            if file_creation_time > initial_time:
                created_files.append(file)
    return created_files

def show_pdb(pdb_file):
    traj = pt.load(pdb_file)
    view = nv.show_pytraj(traj)
    view._remote_call("setSize", target="Widget", args=["800px", "800px"])
    display(view)

def download_pdb(link,filename):
    urllib.request.urlretrieve(link, filename)
    print("File {} has been successfully downloaded".format(filename))

def download_sasdata(link,filename):
    return

def saxs_profile(pdb_files, core="foxs"):
    if isinstance(pdb_files, str):
        pdb_files = [pdb_files]  # Convert single file to a list

    fig, ax = plt.subplots(figsize=(9.5, 6))  # Create the initial figure and axes

    # Dictionary to store the visibility state of each plot
    plot_visibility = {pdb_file: True for pdb_file in pdb_files}

    def update_plot(scale, plot_title):
        ax.clear()
        scale_x, scale_y = scale.split('/')

        handles = []
        labels = []

        for pdb_file in pdb_files:
            if plot_visibility[pdb_file]:
                if core == "pepsisaxs" or core == "both":
                    cmd = ["Pepsi-SAXS", pdb_file]
                    cmd = " ".join(cmd)
                    cmd = f"{cmd} > {pdb_file.removesuffix('.pdb')}_pepsisaxs_output.txt 2>&1"
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    output_file = pdb_file.removesuffix(".pdb") + ".out"
                    skip_rows = 6  # Number of rows to skip in Pepsi-SAXS output
                    names = ["q", "int", "Iat", "Iev", "Ihs", "AatAev", "AatAhs", "AevAhs"]
                    saxs_output = pd.read_table(output_file, sep="\s+", skiprows=skip_rows, names=names, comment='#')
                    line, = ax.plot(saxs_output.q, saxs_output.int, linewidth=2)
                    handles.append(line)
                    labels.append(pdb_file + " (Pepsi-SAXS)")

                if core == "foxs" or core == "both":
                    cmd = ["foxs", pdb_file]
                    cmd = " ".join(cmd)
                    cmd = f"{cmd} > {pdb_file.removesuffix('.pdb')}_foxs_output.txt 2>&1"
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    output_file = pdb_file + ".dat"
                    names = ["q", "int", "err"]
                    saxs_output = pd.read_table(output_file, sep="\s+", names=names, comment='#')
                    line, = ax.plot(saxs_output.q, saxs_output.int, linewidth=2)
                    handles.append(line)
                    labels.append(pdb_file + " (FoXS)")

        ax.set_title(plot_title)
        ax.set_yscale(scale_y)
        ax.set_xscale(scale_x)
        ax.set_xlabel("Q ($\AA^{-1}$)")
        ax.set_ylabel("Intensity")

        ax.legend(handles=handles, labels=labels)
        fig.canvas.draw()  # Redraw the figure

    def update_visibility(change):
        plot_visibility[change.owner.description] = change.new
        update_plot(scale_dropdown.value, title_input.value)

    def save_plot(_):
        filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
        save_button = widgets.Button(description='Save', button_style='success')
        cancel_button = widgets.Button(description='Cancel', button_style='danger')
        file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

        def save_dialog():
            display(file_dialog)

        def save_button_clicked(_):
            filename = filename_widget.value
            fig.savefig(filename, format='png')
            print("Plot saved as", filename)
            file_dialog.close()

        def cancel_button_clicked(_):
            file_dialog.close()

        save_button.on_click(save_button_clicked)
        cancel_button.on_click(cancel_button_clicked)
        save_dialog()

    checkboxes = []  # Store checkboxes
    checkbox_container = widgets.VBox(
        layout=widgets.Layout(display='flex', flex_flow='column', align_items='flex-start'))  # Container for checkboxes

    for pdb_file in pdb_files:
        checkbox = widgets.Checkbox(value=True, description=pdb_file, indent=False,  layout=widgets.Layout(width='100px',margin='0 5px 0 0'))
        checkbox.observe(update_visibility, 'value')
        checkboxes.append(checkbox)
        checkbox_container.children = checkboxes  # Set checkboxes as children of the container

    # Create an interactive widget for selecting the scale
    scale_dropdown = widgets.Dropdown(options=['linear/linear', 'linear/log', 'log/log'], value='linear/linear', description='Scale:')
    title_input = widgets.Text(value='SAXS Profiles', description='Title:', layout=widgets.Layout(width='300px'))
    # Display the interactive widget
    interact(update_plot, scale=scale_dropdown, plot_title=title_input)
    display(checkbox_container)  # Display the checkbox container
    display_button = widgets.Button(description="Save As...")
    display_button.on_click(save_plot)
    display(display_button)

    # Remove the "Save" icon from the toolbar
    toolbar = fig.canvas.toolbar
    for toolitem in toolbar.toolitems:
        if toolitem[0] == 'Download':
            toolbar.toolitems.remove(toolitem)
            break

def sans_profile(pdb_files, deut_level=[0], d2o_level=[0],exchange=[0]):
    if isinstance(pdb_files, str):
        pdb_files = [pdb_files]  # Convert single file to a list

    fig, ax = plt.subplots(figsize=(9.5, 6))  # Create the initial figure and axes

    # Convert deut_level and d2o_level to lists if they are not already
    deut_level = [deut_level] if not isinstance(deut_level, list) else deut_level
    d2o_level = [d2o_level] if not isinstance(d2o_level, list) else d2o_level
    exchange = [exchange] if not isinstance(exchange, list) else exchange

    # Calculate the total number of profiles
    total_profiles = len(pdb_files) * len(deut_level) * len(d2o_level) * len(exchange)

    # Dictionary to store the visibility state of each plot
    plot_visibility = {pdb_file: {deut: {d2o: {exch: True for exch in exchange} for d2o in d2o_level} for deut in deut_level} for pdb_file in pdb_files}

    def update_plot(scale, plot_title):
        ax.clear()  # Clear the previous plot
        scale_x, scale_y = scale.split('/')

        handles = []  # Store handles for legend
        labels = []  # Store labels for legend

        for pdb_file in pdb_files:
            for deut in deut_level:
                for d2o in d2o_level:
                    for exch in exchange:
                        if plot_visibility[pdb_file][deut][d2o][exch]:
                            cmd = ["Pepsi-SANS ", pdb_file, "--deut", str(deut), "--d2o", str(d2o),"--exchange", str(exch)]
                            deut_level_str = str(int(float(deut) * 100))
                            d2o_level_str = str(int(float(d2o) * 100))
                            exch_level_str = str(int(float(exch)*100))
                            cmd.extend(["-o",pdb_file.removesuffix(".pdb")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".out"])
                            cmd = " ".join(cmd)
                            cmd = f"{cmd} > {pdb_file.removesuffix('.pdb')}_fit_pepsisans_output.txt 2>&1"
                            process = subprocess.Popen(cmd, shell=True)
                            process.wait()
                            pepsisans_output = pd.read_table(pdb_file.removesuffix(".pdb")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".out",sep="\s+",skiprows=6,names=["q", "int","Iat","Iev","Ihs","AatAev","AatAhs","AevAhs"])
                            line, = ax.plot(pepsisans_output.q, pepsisans_output.int, linewidth=4)
                            handles.append(line)
                            labels.append(f"{pdb_file} (Deut: {deut}, D2O: {d2o}, Exchange: {exch})")

        ax.set_title(plot_title)
        ax.set_yscale(scale_y)
        ax.set_xscale(scale_x)
        ax.set_xlabel("Q ($\AA^{-1}$)")
        ax.set_ylabel("Intensity")

        ax.legend(handles=handles, labels=labels)
        fig.canvas.draw()  # Redraw the figure

    def update_visibility(change):
        pdb_file = change.owner.pdb_file
        deut = change.owner.deut
        d2o = change.owner.d2o
        exch = change.owner.exch
        plot_visibility[pdb_file][deut][d2o][exch] = change.new
        update_plot(scale_dropdown.value, title_input.value)

    def save_plot(_):
        filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
        save_button = widgets.Button(description='Save', button_style='success')
        cancel_button = widgets.Button(description='Cancel', button_style='danger')
        file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

        def save_dialog():
            display(file_dialog)

        def save_button_clicked(_):
            filename = filename_widget.value
            fig.savefig(filename, format='png')
            print("Plot saved as", filename)
            file_dialog.close()

        def cancel_button_clicked(_):
            file_dialog.close()

        save_button.on_click(save_button_clicked)
        cancel_button.on_click(cancel_button_clicked)
        save_dialog()

    checkboxes = []  # Store checkboxes
    checkbox_container = widgets.VBox(
        layout=widgets.Layout(display='flex', flex_flow='column', align_items='flex-start'))  # Container for checkboxes

    for pdb_file in pdb_files:
        for deut in deut_level:
            for d2o in d2o_level:
                for exch in exchange:
                    checkbox = widgets.Checkbox(value=True, description=f"{pdb_file} - Deut: {deut} - D2O: {d2o} - Exchange: {exch}", indent=False, layout=widgets.Layout(width='300px', margin='0 5px 0 0'))
                    checkbox.observe(update_visibility, 'value')
                    checkbox.pdb_file = pdb_file
                    checkbox.deut = deut
                    checkbox.d2o = d2o
                    checkbox.exch = exch
                    checkboxes.append(checkbox)
                    checkbox_container.children = checkboxes  # Set checkboxes as children of the container

    # Create an interactive widget for selecting the scale
    scale_dropdown = widgets.Dropdown(options=['linear/linear', 'linear/log', 'log/log'], value='linear/linear', description='Scale:')
    title_input = widgets.Text(value='SANS Profiles', description='Title:', layout=widgets.Layout(width='300px'))
    # Display the interactive widget
    interact(update_plot, scale=scale_dropdown, plot_title=title_input)
    display(checkbox_container)  # Display the checkbox container
    display_button = widgets.Button(description="Save As...")
    display_button.on_click(save_plot)
    display(display_button)

    # Remove the "Save" icon from the toolbar
    toolbar = fig.canvas.toolbar
    for toolitem in toolbar.toolitems:
        if toolitem[0] == 'Download':
            toolbar.toolitems.remove(toolitem)
            break

def modelpdb_flex(pdb_file, flex_file, num_iter=100, num_modes=100, rad=0.5, models="all"):
    pdb_file_name = os.path.splitext(pdb_file)[0]  # Remove the ".pdb" suffix

    cmd = ["rrt_sample", pdb_file, flex_file]

    if num_iter != 100:
        cmd.extend(["-i", str(num_iter)])

    if num_modes != 100:
        cmd.extend(["-n", str(num_modes)])

    if models == "all":
        cmd.extend(["-m 1000000"])

    if models == "sep":
        cmd.extend(["-m 1"])

    if rad != 0.5:
        cmd.extend(["-s", str(rad)])

    cmd = " ".join(cmd)
    cmd = f"{cmd} > {pdb_file_name}_rrt_output.txt 2>&1"  # Run the command with output redirection

    # Display progress message
    progress_message = widgets.Label("Modeling in progress...")

    display(progress_message)

    # Run the modeling command
    process = subprocess.Popen(cmd, shell=True)

    # Wait for the modeling process to finish and capture the output
    process.wait()

    # Wait for a short period to ensure the file is written completely
    time.sleep(1)

    # Remove the progress message
    clear_output()

    # Check if the output file exists
    output_file_path = f"{pdb_file_name}_rrt_output.txt"
    if not os.path.exists(output_file_path):
        print("Error: Output file not found")
        return

    # Read the output file
    with open(output_file_path, "r") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            print("Modelling finished")
            print(last_line)
            try:
                # Extract the number after "Done RRT"
                number = int(last_line.split("Done RRT")[1].strip())
                visualize_result(pdb_file_name, number, models)
            except IndexError:
                print("Error: Unable to extract the number from the output line")
        else:
            print("Error: Output file is empty")
    return number

def visualize_result(pdb_file_name, number, models):
    if models == "all":
        # Call show_pdb("nodes1.pdb") to launch the visualization
        show_pdb(f"nodes1.pdb")
    elif models == "sep":
        files = [f"nodes{i}.pdb" for i in range(1, number+1)]
        files.extend([pdb_file_name])
        interact(show_pdb, pdb_file=files)

def modelpdb_nolb(pdb_file,num_iter=500,num_modes=10):
    pdb_file_name = os.path.splitext(pdb_file)[0]  # Remove the ".pdb" suffix

    cmd = ["NOLB ", pdb_file]

    if num_iter != 500:
        cmd.extend(["--nSteps", str(num_iter)])

    if num_modes != 10:
        cmd.extend(["-n", str(num_modes)])

    cmd = " ".join(cmd)
    cmd = f"{cmd} > {pdb_file_name}_nolb_output.txt 2>&1"  # Run the command with output redirection

    # Display progress message
    progress_message = widgets.Label("Modeling in progress...")

    display(progress_message)

    # Run the modeling command
    process = subprocess.Popen(cmd, shell=True)

    # Wait for the modeling process to finish and capture the output
    process.wait()

    # Wait for a short period to ensure the file is written completely
    time.sleep(1)

    # Remove the progress message
    clear_output()

    files = [f"{pdb_file_name}_nlb_{i}.pdb" for i in range(1, num_modes+1)]
    files.extend([pdb_file])
    interact(show_pdb, pdb_file=files)


def fitsaxs(pdb_files, data_file, core="foxs", c1_low=0.99, c1_up=1.05, c2_low=-2, c2_up=4, bg=0, hyd=False, scale=1, int_0=1, neg=False, no_smear=False, hyd_shell=5, conc=1, abs_int=0, bulk_SLD=1e-5):
    if isinstance(pdb_files, str):
        pdb_files = [pdb_files]  # Convert single file to a list

    if core=="foxs":
        cmd = ["foxs"]
        cmd.extend(pdb_files)
        cmd.extend([data_file])

        if c1_low != 0.99:
            cmd.extend(["--min_c1", str(c1_low)])
        if c1_up != 1.05:
            cmd.extend(["--max_c1", str(c1_up)])
        if c2_low != -2:
            cmd.extend(["--min_c2", str(c2_low)])
        if c2_up != 4:
            cmd.extend(["--max_c2", str(c2_up)])
        if bg !=0:
            cmd.extend(["-b", str(bg)])
        if hyd != False:
            cmd.extend(["-h"])

        cmd = " ".join(cmd)
        cmd = f"{cmd} > {data_file}_fit_foxs_output.txt 2>&1"

        progress_message = widgets.Label("Fitting in progress...")
        display(progress_message)

        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        time.sleep(1)
        clear_output()

        values_fit = []
        fit_file = []

        for pdb_file in pdb_files:
            fit_file.append(pdb_file.removesuffix(".pdb") + "_" + data_file.removesuffix(".dat") + ".fit")
            with open(pdb_file.removesuffix(".pdb") + "_" + data_file.removesuffix(".dat") + ".fit","r") as f:
                line = f.readlines()[1].strip()
            index = line.find("Chi^2 = ")
            value = line[index + len("Chi^2 = "):].strip()
            values_fit.append(value)

        data = list(zip(pdb_files, values_fit, fit_file))
        sorted_data = sorted(data, key=lambda x: float(x[1]))
        top_10_data = sorted_data[:10] if len(sorted_data) > 10 else sorted_data
        df = pd.DataFrame(top_10_data, columns=['pdb_file', 'Chi^2', 'fit_file'])
        display(df.style.hide_index())

        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 6))
        with open(data_file, 'r') as file:
            lines = file.readlines()

        lines = [line for line in lines if line.strip()]

        with open(data_file, 'w') as file:
            file.writelines(lines)

        with open(data_file, 'r') as f:
            num_header_rows = 0
            num_footer_rows = 0
            num_blank_header_rows = 0
            num_blank_footer_rows = 0
            data_start = False

            for line in f:
                if not line.strip():
                    if not data_start:
                        num_blank_header_rows += 1
                    else:
                        num_blank_footer_rows += 1
                    continue

                if not data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                        data_start = True
                    except ValueError:
                        num_header_rows += 1

                if data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                    except ValueError:
                        num_footer_rows += 1

            f.seek(0)

            data_to_fit = pd.read_table(
                f,
                sep='\s+',
                skiprows=num_header_rows + num_blank_header_rows,
                skipfooter=num_footer_rows + num_blank_footer_rows,
                engine='python',
                comment="#",
                header=None,
                names=['q', 'int', 'err']
            )

            handles = []
            labels = []

            line = ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')
            handles.append(line)
            labels.append(f"{data_file}")

        # for pdb_file in pdb_files:
        #     foxs_output = pd.read_table(
        #         pdb_file.removesuffix(".pdb") + "_" + data_file.removesuffix(".dat") + ".fit",
        #         sep="\s+",
        #         comment="#",
        #         engine='python',
        #         names=["q", "int", "err", 'fit']
        #     )
        #
        #     line, = ax.plot(foxs_output.q, foxs_output.fit, linewidth=4)
        #     handles.append(line)
        #     labels.append(f"{pdb_file}")
        #
        # ax.legend(handles=handles, labels=labels)
        # ax.set_xlabel("Q ($\AA^{-1}$)")
        # ax.set_ylabel("Intensity")

        # Define checkboxes for top_10 pdb_files
        checkboxes = []
        for i, row in df.iterrows():
            checkbox = widgets.Checkbox(description=row['pdb_file'], value=True)
            checkboxes.append(checkbox)

        # Define scale dropdown and title input
        scale_dropdown = widgets.Dropdown(
            options=['linear/linear', 'linear/log', 'log/log'],
            value='linear/linear',
            description='Scale:'
        )
        title_input = widgets.Text(value='Fitting with FoXS', description='Title:', layout=widgets.Layout(width='300px'))

        vbox = VBox([*checkboxes, scale_dropdown, title_input])


        def update_plot(change):
            ax.clear()
            ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')

            new_handles = [handles[0]]  # Include the initial errorbar plot handle
            new_labels = [labels[0]]  # Include the initial data file label

            for i, checkbox in enumerate(checkboxes):
                if checkbox.value:
                    pdb_file = df.loc[i, 'pdb_file']
                    foxs_output = pd.read_table(
                        pdb_file.removesuffix(".pdb") + "_" + data_file.removesuffix(".dat") + ".fit",
                        sep="\s+",
                        comment="#",
                        engine='python',
                        names=["q", "int", "err", 'fit']
                    )
                    line, = ax.plot(foxs_output.q, foxs_output.fit, linewidth=4)
                    new_handles.append(line)
                    new_labels.append(pdb_file)

            ax.legend(handles=new_handles, labels=new_labels)
            ax.set_xlabel("Q ($\AA^{-1}$)")
            ax.set_ylabel("Intensity")
            ax.set_title(title_input.value)
            ax.set_yscale(scale_dropdown.value.split('/')[1])
            ax.set_xscale(scale_dropdown.value.split('/')[0])

            plt.show()

        # Update the plot when the checkboxes, scale dropdown, or title input change
        for checkbox in checkboxes:
            checkbox.observe(update_plot, 'value')

        scale_dropdown.observe(update_plot, 'value')
        title_input.observe(update_plot, 'value')
        def save_plot(_):
            filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
            save_button = widgets.Button(description='Save', button_style='success')
            cancel_button = widgets.Button(description='Cancel', button_style='danger')
            file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

            def save_dialog():
                display(file_dialog)

            def save_button_clicked(_):
                filename = filename_widget.value
                fig.savefig(filename, format='png')
                print("Plot saved as", filename)
                file_dialog.close()

            def cancel_button_clicked(_):
                file_dialog.close()

            save_button.on_click(save_button_clicked)
            cancel_button.on_click(cancel_button_clicked)
            save_dialog()
        # Combine checkboxes, scale dropdown, and title input vertically using VBox
        display_button = widgets.Button(description="Save As...")
        display_button.on_click(save_plot)
        display(display_button)
        # Show the initial plot
        update_plot(None)
        display(vbox)
        toolbar = fig.canvas.toolbar
        for toolitem in toolbar.toolitems:
            if toolitem[0] == 'Download':
                toolbar.toolitems.remove(toolitem)
                break
    elif core == "pepsisaxs":
        fit_file = []
        values_fit =[]
        progress_message = widgets.Label(f"Fitting in progress...")
        display(progress_message)
        for pdb_file in pdb_files:
            cmd = ["Pepsi-SAXS", pdb_file, data_file]
            if bg != 0:
                cmd.extend(["--cstFactor", str(bg)])
            if hyd != False:
                cmd.extend(["-hyd"])
            if scale != 1:
                cmd.extend(["--scaleFactor", str(scale)])
            if int_0 != 1:
                cmd.extend(["--I0", str(int_0)])
            if neg != False:
                cmd.extend(["--neg"])
            if no_smear != False:
                cmd.extend(["--noSmearing"])
            if hyd_shell != 5:
                cmd.extend(["--dro", str(hyd_shell)])
            if conc != 1:
                cmd.extend(["--conc", str(conc)])
            if abs_int != 0:
                cmd.extend(["--absFit", str(abs_int)])
            if bulk_SLD != 1e-5:
                cmd.extend(["--bulkSLD", str(bulk_SLD)])
            cmd = " ".join(cmd)
            cmd = f"{cmd} > {pdb_file.removesuffix('.pdb')}_fit_pepsisaxs_output.txt 2>&1"
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            fit_file.append(pdb_file.removesuffix(".pdb") + "-" + data_file.removesuffix(".dat") + ".fit")
            with open(fit_file[-1], "r") as f:
                line = f.readlines()[4].strip()
            index = line.find("Chi2: ")
            value = line[index + len("Chi2: "):].strip()
            values_fit.append(value)
        clear_output()
        data = list(zip(pdb_files, values_fit, fit_file))
        sorted_data = sorted(data, key=lambda x: float(x[1]))
        top_10_data = sorted_data[:10] if len(sorted_data) > 10 else sorted_data
        df = pd.DataFrame(top_10_data, columns=['pdb_file', 'Chi^2', 'fit_file'])
        display(df.style.hide_index())

        # Create plot
        fig, ax = plt.subplots(figsize=(9.5, 6))
        with open(data_file, 'r') as file:
            lines = file.readlines()
        lines = [line for line in lines if line.strip()]
        with open(data_file, 'w') as file:
            file.writelines(lines)
        with open(data_file, 'r') as f:
            num_header_rows = 0
            num_footer_rows = 0
            num_blank_header_rows = 0
            num_blank_footer_rows = 0
            data_start = False

            for line in f:
                if not line.strip():
                    if not data_start:
                        num_blank_header_rows += 1
                    else:
                        num_blank_footer_rows += 1
                    continue

                if not data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                        data_start = True
                    except ValueError:
                        num_header_rows += 1

                if data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                    except ValueError:
                        num_footer_rows += 1

            f.seek(0)

            data_to_fit = pd.read_table(
                f,
                sep='\s+',
                skiprows=num_header_rows + num_blank_header_rows,
                skipfooter=num_footer_rows + num_blank_footer_rows,
                engine='python',
                comment="#",
                header=None,
                names=['q', 'int', 'err']
            )

            handles = []
            labels = []

            line = ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')
            handles.append(line)
            labels.append(f"{data_file}")

        # for pdb_file in pdb_files:
        #     pepsi_output = pd.read_table(
        #         pdb_file.removesuffix(".pdb") + "_" + data_file.removesuffix(".dat") + ".fit",
        #         sep="\s+",
        #         comment="#",
        #         engine='python',
        #         names=["q", "int", "err", 'fit']
        #     )
        #
        #     line, = ax.plot(pepsi_output.q, pepsi_output.fit, linewidth=4)
        #     handles.append(line)
        #     labels.append(f"{pdb_file}")
        #
        # ax.legend(handles=handles, labels=labels)
        # ax.set_xlabel("Q ($\AA^{-1}$)")
        # ax.set_ylabel("Intensity")
        #
        # # Define checkboxes for top_10 pdb_files
        checkboxes = []
        for i, row in df.iterrows():
            checkbox = widgets.Checkbox(description=row['pdb_file'], value=True)
            checkboxes.append(checkbox)

        # Define scale dropdown and title input
        scale_dropdown = widgets.Dropdown(
            options=['linear/linear', 'linear/log', 'log/log'],
            value='linear/linear',
            description='Scale:'
        )
        title_input = widgets.Text(value='Fitting with Pepsi-SAXS', description='Title:', layout=widgets.Layout(width='300px'))
        def save_plot(_):
            filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
            save_button = widgets.Button(description='Save', button_style='success')
            cancel_button = widgets.Button(description='Cancel', button_style='danger')
            file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

            def save_dialog():
                display(file_dialog)

            def save_button_clicked(_):
                filename = filename_widget.value
                fig.savefig(filename, format='png')
                print("Plot saved as", filename)
                file_dialog.close()

            def cancel_button_clicked(_):
                file_dialog.close()

            save_button.on_click(save_button_clicked)
            cancel_button.on_click(cancel_button_clicked)
            save_dialog()
        # Combine checkboxes, scale dropdown, and title input vertically using VBox
        display_button = widgets.Button(description="Save As...")
        display_button.on_click(save_plot)
        display(display_button)
        # Combine checkboxes, scale dropdown, and title input vertically using VBox
        vbox = VBox([*checkboxes, scale_dropdown, title_input])

        def update_plot(change):
            ax.clear()
            ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')

            new_handles = [handles[0]]  # Include the initial errorbar plot handle
            new_labels = [labels[0]]  # Include the initial data file label

            for i, checkbox in enumerate(checkboxes):
                if checkbox.value:
                    pdb_file = df.loc[i, 'pdb_file']
                    pepsi_output = pd.read_table(
                        pdb_file.removesuffix(".pdb") + "-" + data_file.removesuffix(".dat") + ".fit",
                        sep="\s+",
                        comment="#",
                        engine='python',
                        names=["q", "int", "err", 'fit']
                    )
                    line, = ax.plot(pepsi_output.q, pepsi_output.fit, linewidth=4)
                    new_handles.append(line)
                    new_labels.append(pdb_file)

            ax.legend(handles=new_handles, labels=new_labels)
            ax.set_xlabel("Q ($\AA^{-1}$)")
            ax.set_ylabel("Intensity")
            ax.set_title(title_input.value)
            ax.set_yscale(scale_dropdown.value.split('/')[1])
            ax.set_xscale(scale_dropdown.value.split('/')[0])

            plt.show()


        for checkbox in checkboxes:
            checkbox.observe(update_plot, 'value')

        scale_dropdown.observe(update_plot, 'value')
        title_input.observe(update_plot, 'value')
        update_plot(None)
        display(vbox)
        toolbar = fig.canvas.toolbar
        for toolitem in toolbar.toolitems:
            if toolitem[0] == 'Download':
                toolbar.toolitems.remove(toolitem)
                break

def fitsans(pdb_files, data_file, deut_level=[0], d2o_level=[0], exchange=[0], bg=0, hyd=False, scale=1, neg=False, no_smear=False, hyd_shell=5, conc=1, abs_int=0, bulk_SLD=1e-5):
    if isinstance(pdb_files, str): # think of int_0? --> the same for multimodelling
        pdb_files = [pdb_files]
    fit_file = []
    values_fit =[]
    deut_level = [deut_level] if not isinstance(deut_level, list) else deut_level
    d2o_level = [d2o_level] if not isinstance(d2o_level, list) else d2o_level
    exchange = [exchange] if not isinstance(exchange, list) else exchange
    progress_message = widgets.Label(f"Fitting in progress...")
    display(progress_message)
    for pdb_file in pdb_files:
        for deut in deut_level:
            for d2o in d2o_level:
                for exch in exchange:
                    cmd = ["Pepsi-SANS ", pdb_file, data_file, " --deut ", str(deut), " --d2o ", str(d2o)," --exchange ", str(exch)]
                    if bg != 0:
                        cmd.extend(["--cstFactor",str(bg)])
                    if hyd != False:
                        cmd.extend(["-hyd"])
                    if scale != 1:
                        cmd.extend(["--scaleFactor",str(scale)])
                    if neg != False:
                        cmd.extend(["--neg"])
                    if no_smear != False:
                        cmd.extend(["--noSmearing"])
                    if hyd_shell != 5:
                        cmd.extend(["--dro",str(hyd_shell)])
                    if conc != 1:
                        cmd.extend(["--conc",str(conc)])
                    if abs_int != 0:
                        cmd.extend(["--absFit",str(abs_int)])
                    if bulk_SLD != 1e-5:
                        cmd.extend(["--bulkSLD",str(bulk_SLD)])
                    deut_level_str = str(int(float(deut) * 100))
                    d2o_level_str = str(int(float(d2o) * 100))
                    exch_level_str = str(int(float(exch)*100))
                    cmd.extend(["-o",pdb_file.removesuffix(".pdb")+"-"+data_file.removesuffix(".dat")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".fit"])
                    cmd = " ".join(cmd)
                    cmd = f"{cmd} > {pdb_file.removesuffix('.pdb')}_fit_pepsisans_output.txt 2>&1"
                    process = subprocess.Popen(cmd, shell=True)
                    process.wait()
                    fit_file.append(pdb_file.removesuffix(".pdb")+"-"+data_file.removesuffix(".dat")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".fit")
                    with open(fit_file[-1], "r") as f:
                            line = f.readlines()[4].strip()
                    index = line.find("Chi2: ")
                    value = line[index + len("Chi2: "):].strip()
                    values_fit.append(value)
    clear_output()
    data = list(zip(values_fit, fit_file))
    sorted_data = sorted(data, key=lambda x: float(x[0]))
    top_10_data = sorted_data[:10] if len(sorted_data) > 10 else sorted_data
    df = pd.DataFrame(top_10_data, columns=['Chi^2', 'fit_file'])
    df.insert(0,'pdb_file',[t[1].split("-"+data_file.removesuffix(".dat"))[0]+".pdb" for t in top_10_data])
    df.insert(2,'Deut',[float(t[1].split("_d2o")[1].split("_exch")[0])/100 for t in top_10_data])
    df.insert(3,'D2O',[float(t[1].split("_d2o")[1].split("_exch")[0])/100 for t in top_10_data])
    df.insert(4,'H-exchange',[float(t[1].split("_exch")[1].split(".fit")[0])/100 for t in top_10_data])
    display(df.style.hide_index())
        # Create plot
    fig, ax = plt.subplots(figsize=(9.5, 6))
    with open(data_file, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if line.strip()]
    with open(data_file, 'w') as file:
        file.writelines(lines)
    with open(data_file, 'r') as f:
        num_header_rows = 0
        num_footer_rows = 0
        num_blank_header_rows = 0
        num_blank_footer_rows = 0
        data_start = False

        for line in f:
            if not line.strip():
                if not data_start:
                    num_blank_header_rows += 1
                else:
                    num_blank_footer_rows += 1
                continue

            if not data_start:
                try:
                    _ = [float(x) for x in line.split()]
                    data_start = True
                except ValueError:
                    num_header_rows += 1

            if data_start:
                try:
                    _ = [float(x) for x in line.split()]
                except ValueError:
                    num_footer_rows += 1

        f.seek(0)

        data_to_fit = pd.read_table(
            f,
            sep='\s+',
            skiprows=num_header_rows + num_blank_header_rows,
            skipfooter=num_footer_rows + num_blank_footer_rows,
            engine='python',
            comment="#"
        )
        if data_to_fit.shape[1] == 4:
            shape = 4;
        else:
            shape = 3;
    if shape == 3:
        data_to_fit = pd.read_table(
                data_file,
                sep='\s+',
                skiprows=num_header_rows + num_blank_header_rows,
                skipfooter=num_footer_rows + num_blank_footer_rows,
                engine='python',
                comment="#",
                names=['q','int','err'])
    else:
        data_to_fit = pd.read_table(
            data_file,
            sep='\s+',
            skiprows=num_header_rows + num_blank_header_rows,
            skipfooter=num_footer_rows + num_blank_footer_rows,
            engine='python',
            comment="#",
            names=['q','int','err','res'])
    handles = []
    labels = []
    line = ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')
    handles.append(line)
    labels.append(f"{data_file}")
    checkboxes = []
    for i, row in df.iterrows():
        checkbox = widgets.Checkbox(description=f"{df.loc[i,'pdb_file']} (Deut: {df.loc[i,'Deut']}, D2O: {df.loc[i,'D2O']}, H-ex: {df.loc[i,'H-exchange']})", value=True,layout=widgets.Layout(width='500px'))
        checkboxes.append(checkbox)

    # Define scale dropdown and title input
    scale_dropdown = widgets.Dropdown(
        options=['linear/linear', 'linear/log', 'log/log'],
        value='linear/linear',
        description='Scale:',
        layout=widgets.Layout(flex='start')
    )
    title_input = widgets.Text(value='Fitting with Pepsi-SANS', description='Title:', layout=widgets.Layout(width='300px',align_items='flex-start'))
    def save_plot(_):
        filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
        save_button = widgets.Button(description='Save', button_style='success')
        cancel_button = widgets.Button(description='Cancel', button_style='danger')
        file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

        def save_dialog():
            display(file_dialog)

        def save_button_clicked(_):
            filename = filename_widget.value
            fig.savefig(filename, format='png')
            print("Plot saved as", filename)
            file_dialog.close()

        def cancel_button_clicked(_):
            file_dialog.close()

        save_button.on_click(save_button_clicked)
        cancel_button.on_click(cancel_button_clicked)
        save_dialog()
    # Combine checkboxes, scale dropdown, and title input vertically using VBox
    display_button = widgets.Button(description="Save As...")
    display_button.on_click(save_plot)
    display(display_button)
    # Combine checkboxes, scale dropdown, and title input vertically using VBox
    vbox = VBox([*checkboxes, scale_dropdown, title_input],layout=widgets.Layout(flex='start'))

    def update_plot(change):
        ax.clear()
        ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')

        new_handles = [handles[0]]  # Include the initial errorbar plot handle
        new_labels = [labels[0]]  # Include the initial data file label

        for i, checkbox in enumerate(checkboxes):
            if checkbox.value:
                if shape == 3:
                    fit_file = df.loc[i, 'fit_file']
                    pepsi_output = pd.read_table(
                        fit_file,
                        sep="\s+",
                        comment="#",
                        engine='python',
                        names=["q", "int", "err", 'fit']
                    )
                    line, = ax.plot(pepsi_output.q, pepsi_output.fit, linewidth=4)
                    new_handles.append(line)
                    new_labels.append(f"{df.loc[i,'pdb_file']} (Deut: {df.loc[i,'Deut']}, D2O: {df.loc[i,'D2O']}, H-ex: {df.loc[i,'H-exchange']})")

                if shape == 4:
                    fit_file = df.loc[i, 'fit_file']
                    pepsi_output = pd.read_table(
                        fit_file,
                        sep="\s+",
                        comment="#",
                        engine='python',
                        names=["q", "int", "err", "res","fit"]
                    )
                    line, = ax.plot(pepsi_output.q, pepsi_output.fit, linewidth=4)
                    new_handles.append(line)
                    new_labels.append(f"{df.loc[i,'pdb_file']} (Deut: {df.loc[i,'Deut']}, D2O: {df.loc[i,'D2O']}, H-ex: {df.loc[i,'H-exchange']})")
        ax.legend(handles=new_handles, labels=new_labels)
        ax.set_xlabel("Q ($\AA^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.set_title(title_input.value)
        ax.set_yscale(scale_dropdown.value.split('/')[1])
        ax.set_xscale(scale_dropdown.value.split('/')[0])

        plt.show()


    for checkbox in checkboxes:
        checkbox.observe(update_plot, 'value')

    scale_dropdown.observe(update_plot, 'value')
    title_input.observe(update_plot, 'value')
    update_plot(None)
    display(vbox)
    toolbar = fig.canvas.toolbar
    for toolitem in toolbar.toolitems:
        if toolitem[0] == 'Download':
            toolbar.toolitems.remove(toolitem)
            break

def multimodelfit(pdb_files, data_file, type="saxs", ensemble_size=10, bestK=1000, chi_perc=0.3, chi=0, min_weight=0.05, max_q=0.5, c1_low=0.99, c1_up=1.05, c2_low=-0.5, c2_up=2, multimodel=1, bg=0, nnls=False,
                  deut_level=[0], d2o_level=[0], exchange=[0], conc=1, abs_int=0, hyd=False, bulk_SLD=1e-5, no_smear=False, scale=1, neg=False, hyd_shell=5):
    if isinstance(pdb_files, str):
        pdb_files = [pdb_files]
    with open(data_file, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if line.strip()]
    with open(data_file, 'w') as file:
        file.writelines(lines)
    if type=="sans":
        sans_profiles = [];
        deut_level = [deut_level] if not isinstance(deut_level, list) else deut_level
        d2o_level = [d2o_level] if not isinstance(d2o_level, list) else d2o_level
        exchange = [exchange] if not isinstance(exchange, list) else exchange
        progress_message = widgets.Label(f"Preparing SANS profiles...")
        display(progress_message)
        for pdb_file in pdb_files:
            for deut in deut_level:
                for d2o in d2o_level:
                    for exch in exchange:
                        cmd = ["Pepsi-SANS ", pdb_file, data_file, " --deut ", str(deut), " --d2o ", str(d2o)," --exchange ", str(exch)]
                        if bg != 0:
                            cmd.extend(["--cstFactor",str(bg)])
                        if hyd != False:
                            cmd.extend(["-hyd"])
                        if scale != 1:
                            cmd.extend(["--scaleFactor",str(scale)])
                        if neg != False:
                            cmd.extend(["--neg"])
                        if no_smear != False:
                            cmd.extend(["--noSmearing"])
                        if hyd_shell != 5:
                            cmd.extend(["--dro",str(hyd_shell)])
                        if conc != 1:
                            cmd.extend(["--conc",str(conc)])
                        if abs_int != 0:
                            cmd.extend(["--absFit",str(abs_int)])
                        if bulk_SLD != 1e-5:
                            cmd.extend(["--bulkSLD",str(bulk_SLD)])
                        deut_level_str = str(int(float(deut) * 100))
                        d2o_level_str = str(int(float(d2o) * 100))
                        exch_level_str = str(int(float(exch)*100))
                        cmd.extend(["-o",pdb_file.removesuffix(".pdb")+"-"+data_file.removesuffix(".dat")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".fit"])
                        cmd = " ".join(cmd)
                        cmd = f"{cmd} >/dev/null 2>&1"
                        process = subprocess.Popen(cmd, shell=True)
                        process.wait()
                        sans_profiles.append(pdb_file.removesuffix(".pdb")+"-"+data_file.removesuffix(".dat")+"_deut"+deut_level_str+"_d2o"+d2o_level_str+"_exch"+exch_level_str+".fit")
        fit_file=[];
        with open(data_file, 'r') as f:
            num_header_rows = 0
            num_footer_rows = 0
            num_blank_header_rows = 0
            num_blank_footer_rows = 0
            data_start = False

            for line in f:
                if not line.strip():
                    if not data_start:
                        num_blank_header_rows += 1
                    else:
                        num_blank_footer_rows += 1
                    continue

                if not data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                        data_start = True
                    except ValueError:
                        num_header_rows += 1

                if data_start:
                    try:
                        _ = [float(x) for x in line.split()]
                    except ValueError:
                        num_footer_rows += 1

            f.seek(0)

            data_to_fit = pd.read_table(
                f,
                sep='\s+',
                skiprows=num_header_rows + num_blank_header_rows,
                skipfooter=num_footer_rows + num_blank_footer_rows,
                engine='python',
                comment="#"
            )
            if data_to_fit.shape[1] == 4:
                shape = 4;
                data_to_fit=pd.read_table(data_file,sep='\s+',
                                          skiprows=num_header_rows + num_blank_header_rows,
                                          skipfooter=num_footer_rows + num_blank_footer_rows,
                                          engine='python',
                                          comment="#", names=['q','int','err','res'])
                data_to_fit.drop(columns=['res'],inplace=True)
                data_to_fit.to_csv(data_file.removesuffix(".dat")+"_3col.dat",index=False,sep="\t")
                file = open("profiles", "w")
                for sans_p in sans_profiles:
                    pepsisans_output = pd.read_table(sans_p, sep="\s+", comment='#',names = ['q','int','err','res','fit'])
                    pepsisans_output.drop(columns=["res"], inplace=True)
                    pepsisans_output.to_csv(sans_p.removesuffix(".fit")+"_3col.fit",index=False,sep="\t")
                    file.write(sans_p.removesuffix(".fit")+"_3col.fit"+ "\n")
                file.close()
            else:
                file = open("profiles", "w")
                for sans_p in sans_profiles:
                    file.write(sans_p + "\n")
                file.close()
                data_to_fit = pd.read_table(data_file,sep='\s+',
                                            skiprows=num_header_rows + num_blank_header_rows,
                                            skipfooter=num_footer_rows + num_blank_footer_rows,
                                            engine='python',
                                            comment="#", names=['q','int','err'])
        clear_output()
        progress_message = widgets.Label(f"Multifit in progress...")
        display(progress_message)
        cmd = ["multi_foxs", data_file]
        if ensemble_size != 10:
            cmd.extend(["-s",str(ensemble_size)])
        if bestK != 1000:
            cmd.extend(["-k",str(bestK)])
        if chi_perc != 0.3:
            cmd.extend(["-t",str(chi_perc)])
        if chi != 0:
            cmd.extend(["-c",str(chi)])
        if min_weight != 0.05:
            cmd.extend(["-w",str(min_weight)])
        if max_q != 0.5:
            cmd.extend(["-q",str(max_q)])
        if c1_low != 0.99:
            cmd.extend(["--min_c1",str(c1_low)])
        if c1_up != 1.05:
            cmd.extend(["--max_c1",str(c1_up)])
        if c2_low != -0.5:
            cmd.extend(["--min_c2",str(c2_low)])
        if c2_up != 2:
            cmd.extend(["--max_c2",str(c2_up)])
        if multimodel != 1:
            cmd.extend(["-m",str(multimodel)])
        if bg != 0:
            cmd.extend(["-b",str(bg)])
        if nnls != False:
            cmd.extend(["-n"])
        cmd.extend(["-p profiles"])
        cmd = " ".join(cmd)
        cmd = f"{cmd} >/dev/null 2>&1"
        #process = subprocess.Popen(cmd, shell=True)
        #process.wait()
        created_files = get_created_files(cmd)
        clear_output()
        display(created_files)

    if type=="saxs":
        progress_message = widgets.Label(f"Multifit in progress...")
        display(progress_message)
        cmd = ["multi_foxs"]
        cmd.extend(pdb_files)
        cmd.extend([data_file])
        if ensemble_size != 10:
            cmd.extend(["-s",str(ensemble_size)])
        if bestK != 1000:
            cmd.extend(["-k",str(bestK)])
        if chi_perc != 0.3:
            cmd.extend(["-t",str(chi_perc)])
        if chi != 0:
            cmd.extend(["-c",str(chi)])
        if min_weight != 0.05:
            cmd.extend(["-w",str(min_weight)])
        if max_q != 0.5:
            cmd.extend(["-q",str(max_q)])
        if c1_low != 0.99:
            cmd.extend(["--min_c1",str(c1_low)])
        if c1_up != 1.05:
            cmd.extend(["--max_c1",str(c1_up)])
        if c2_low != -0.5:
            cmd.extend(["--min_c2",str(c2_low)])
        if c2_up != 2:
            cmd.extend(["--max_c2",str(c2_up)])
        if multimodel != 1:
            cmd.extend(["-m",str(multimodel)])
        if bg != 0:
            cmd.extend(["-b",str(bg)])
        if nnls != False:
            cmd.extend(["-n"])
        cmd = " ".join(cmd)
        cmd = f"{cmd} >/dev/null 2>&1"
        #process = subprocess.Popen(cmd, shell=True)
        #process.wait()
        created_files = get_created_files(cmd)
        clear_output()
    fitlist = [file for file in created_files if file.endswith('.fit') and file.startswith('multi_state_model')]
    values=[]
    for i in range(len(fitlist)):
        with open(fitlist[i],"r") as f:
            line = f.readlines()[1].strip()
        index = line.find("Chi^2 = ")
        value = line[index+len("Chi^2 = "):].strip()
        values.append(value)
    data = list(zip(values, fitlist))
    sorted_data = sorted(data, key=lambda x: float(x[0]))
    top_10_data = sorted_data[:10] if len(sorted_data) > 10 else sorted_data
    df = pd.DataFrame(top_10_data, columns=['Chi^2', 'fit_file'])
    contributing_pdbs = []
    weights = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        fit_file = row['fit_file']
        match = re.match(r"multi_state_model_(\d+)_(\d+)_1.fit", fit_file)

        if match:
            number1 = int(match.group(1))
            number2 = int(match.group(2))

            ensemble_file_name = f"ensembles_size_{number1}.txt"

            with open(ensemble_file_name, "r") as ensemble_file:
                lines = ensemble_file.readlines()

                for line_num, line in enumerate(lines):
                    if line.startswith(f"{number2} |"):
                        pdb_lines = [lines[line_num + i].strip() for i in range(1, number1 + 1)]
                        pdb_info = [re.search(r'\| (.+?) \|', pdb_line).group(1) for pdb_line in pdb_lines]
                        pdb_files = [pdb_line.split('|')[2].strip().split()[0] for pdb_line in pdb_lines]
                        pdb_numbers = [float(info.split()[0]) for info in pdb_info]

                        contributing_pdbs.append(', '.join(pdb_files))
                        weights.append(', '.join(map(str, pdb_numbers)))

    # Add the new columns to the DataFrame
    df['Contributing PDB file(s)'] = contributing_pdbs
    df['Weight(s)'] = weights
    #df.insert(1,'Contributing PDB file(s)',[t[1].split("-"+data_file.removesuffix(".dat"))[0]+".pdb" for t in top_10_data])
    display(df)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    handles = []
    labels = []
    with open(data_file, 'r') as f:
        num_header_rows = 0
        num_footer_rows = 0
        num_blank_header_rows = 0
        num_blank_footer_rows = 0
        data_start = False

        for line in f:
            if not line.strip():
                if not data_start:
                    num_blank_header_rows += 1
                else:
                    num_blank_footer_rows += 1
                continue

            if not data_start:
                try:
                    _ = [float(x) for x in line.split()]
                    data_start = True
                except ValueError:
                    num_header_rows += 1

            if data_start:
                try:
                    _ = [float(x) for x in line.split()]
                except ValueError:
                    num_footer_rows += 1

        f.seek(0)

    data_to_fit = pd.read_table(
        data_file,
        sep='\s+',
        skiprows=num_header_rows + num_blank_header_rows,
        skipfooter=num_footer_rows + num_blank_footer_rows,
        engine='python',
        comment="#",
        names=['q','int','err'])
    line = ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')
    handles.append(line)
    labels.append(f"{data_file}")
    checkboxes_and_buttons = []
    checkboxes = []
    for i, row in df.iterrows():
        checkbox = widgets.Checkbox(description=f"{df.loc[i,'fit_file']}", value=True,layout=widgets.Layout(width='500px'))
        button_show = widgets.Button(description="Show PDBs")
        checkboxes_and_buttons.append((checkbox,button_show))
        checkboxes.append(checkbox)
    # Define scale dropdown and title input
    scale_dropdown = widgets.Dropdown(
        options=['linear/linear', 'linear/log', 'log/log'],
        value='linear/linear',
        description='Scale:',
        layout=widgets.Layout(flex='start')
    )
    title_input = widgets.Text(value='Multifit', description='Title:', layout=widgets.Layout(width='300px',align_items='flex-start'))
    def save_plot(_):
        filename_widget = widgets.Text(value='', placeholder='Enter a filename (including extension)', description='Filename:', layout=widgets.Layout(width='400px'))
        save_button = widgets.Button(description='Save', button_style='success')
        cancel_button = widgets.Button(description='Cancel', button_style='danger')
        file_dialog = widgets.HBox([filename_widget, save_button, cancel_button])

        def save_dialog():
            display(file_dialog)

        def save_button_clicked(_):
            filename = filename_widget.value
            fig.savefig(filename, format='png')
            print("Plot saved as", filename)
            file_dialog.close()

        def cancel_button_clicked(_):
            file_dialog.close()

        save_button.on_click(save_button_clicked)
        cancel_button.on_click(cancel_button_clicked)
        save_dialog()
    # Combine checkboxes, scale dropdown, and title input vertically using VBox
    display_button = widgets.Button(description="Save As...")
    display_button.on_click(save_plot)
    display(display_button)
    # Combine checkboxes, scale dropdown, and title input vertically using VBox
    vbox = VBox([*checkboxes, scale_dropdown, title_input],layout=widgets.Layout(flex='start'))

    def update_plot(change):
        ax.clear()
        ax.errorbar(data_to_fit.q, data_to_fit.int, yerr=data_to_fit.err, fmt='o', zorder=-1, c='k')

        new_handles = [handles[0]]  # Include the initial errorbar plot handle
        new_labels = [labels[0]]  # Include the initial data file label

        for i, checkbox in enumerate(checkboxes):
            if checkbox.value:
                fit_file = df.loc[i, 'fit_file']
                multifoxs_output = pd.read_table(
                    fit_file,
                    sep="\s+",
                    comment="#",
                    engine='python',
                    names=["q", "int", "err", 'fit']
                )
                line, = ax.plot(multifoxs_output.q, multifoxs_output.fit, linewidth=4)
                new_handles.append(line)
                new_labels.append(f"{df.loc[i,'fit_file']}")
        ax.legend(handles=new_handles, labels=new_labels)
        ax.set_xlabel("Q ($\AA^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.set_title(title_input.value)
        ax.set_yscale(scale_dropdown.value.split('/')[1])
        ax.set_xscale(scale_dropdown.value.split('/')[0])

        plt.show()


    for checkbox in checkboxes:
        checkbox.observe(update_plot, 'value')

    scale_dropdown.observe(update_plot, 'value')
    title_input.observe(update_plot, 'value')
    update_plot(None)
    display(vbox)
    toolbar = fig.canvas.toolbar
    for toolitem in toolbar.toolitems:
        if toolitem[0] == 'Download':
            toolbar.toolitems.remove(toolitem)
            break
    return df
