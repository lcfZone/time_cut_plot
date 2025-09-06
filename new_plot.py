import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import threading
import queue
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.dates as mdates
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 闰秒
leap_seconds_list = [
    46828800, 78364801, 109900802, 173059203, 252028804, 315187205,
    346723206, 393984007, 425520008, 457056009, 504489610, 551750411,
    599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017
]


# GPS时间转换
def gps_ms_to_bj(ms):
    gps_epoch = datetime(1980, 1, 6, tzinfo=ZoneInfo('UTC'))
    gps_sec = ms / 1000
    leap = sum(1 for ls in leap_seconds_list if ls <= gps_sec)
    seconds_int = int(gps_sec)
    milliseconds = int((gps_sec - seconds_int) * 1000000)
    utc = gps_epoch + timedelta(seconds=seconds_int - leap, microseconds=milliseconds)
    return utc.astimezone(ZoneInfo('Asia/Shanghai'))


# 文件名解析
def parse_filename_to_time(fname):
    base = os.path.splitext(fname)[0]
    ts = base.split('_')[0]
    return gps_ms_to_bj(int(ts))


# 时间区间合并
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + timedelta(seconds=1):
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


# 分析时间段
def analyze_intervals(inp):
    files = sorted(f for f in os.listdir(inp) if f.endswith('.dat'))
    intervals = []
    for fname in files:
        try:
            st = parse_filename_to_time(fname)
            file_path = os.path.join(inp, fname)
            file_size = os.path.getsize(file_path)
            data_len = file_size // 4
            ed = st + timedelta(seconds=(data_len - 1) / 1000)
            intervals.append((st, ed))
        except Exception as ex:
            pass
    return merge_intervals(intervals)


# 文件分组
def group_files_by_intervals(inp, intervals):
    files = sorted(f for f in os.listdir(inp) if f.endswith('.dat'))
    grouped = {interval: [] for interval in intervals}
    for fname in files:
        st = parse_filename_to_time(fname)
        for interval in intervals:
            if interval[0] <= st <= interval[1]:
                grouped[interval].append(fname)
                break
    return grouped


# 文件处理
def process_file_batch(file_batch, input_dir, output_queue):
    batch_arrays = []
    for f in file_batch:
        try:
            file_path = os.path.join(input_dir, f)
            batch_arrays.append(np.fromfile(file_path, dtype=np.int32))
        except Exception as e:
            output_queue.put(('error', f"处理文件 {f} 时出错: {e}\n"))
    return batch_arrays


# 合并数据
def combine_data(inp, out, q, progress_q, time_intervals=None):
    try:
        os.makedirs(out, exist_ok=True)
        if not time_intervals:
            q.put(('update', '分析时间段以分组文件...\n'))
            time_intervals = analyze_intervals(inp)
        grouped_files = group_files_by_intervals(inp, time_intervals)
        total_intervals = len(time_intervals)
        for i, (interval, files) in enumerate(grouped_files.items()):
            if not files:
                continue
            start_time, end_time = interval
            folder_name = os.path.basename(os.path.normpath(inp))
            safe_folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', folder_name)
            combined_file = os.path.join(
                out,
                f'{safe_folder_name}_combined_data_{start_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]}_{end_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]}.txt'
            )
            if os.path.isfile(combined_file):
                q.put(('update', f'发现 {os.path.basename(combined_file)}，跳过合并\n'))
                continue
            batch_size = max(1, min(100, len(files) // 10))
            batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
            all_arrays = []
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                for batch in batches:
                    arrays = process_file_batch(batch, inp, q)
                    all_arrays.extend(arrays)
                    progress = int((i + (batches.index(batch) + 1) / len(batches)) / total_intervals * 100)
                    progress_q.put(progress)
            data = np.concatenate(all_arrays) if all_arrays else np.array([], dtype=int)
            np.savetxt(combined_file, data, fmt='%d')
            q.put(('update', f'合并完成: {os.path.basename(combined_file)}\n'))
        q.put(('complete', '所有时间段合并完成\n'))
    except Exception as ex:
        q.put(('error', f'合并出错: {ex}\n'))
    finally:
        progress_q.put(100)


# 查找合并文件
def find_combined_file(out, cs, ce, folder_name):
    safe_folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', folder_name)
    pattern = rf'^{re.escape(safe_folder_name)}_combined_data_(\d{{8}}_\d{{6}}_\d{{3}})_(\d{{8}}_\d{{6}}_\d{{3}})\.txt$'
    for f in os.listdir(out):
        match = re.match(pattern, f)
        if match:
            start_str, end_str = match.groups()
            start_time = datetime.strptime(start_str, '%Y%m%d_%H%M%S_%f').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
            end_time = datetime.strptime(end_str, '%Y%m%d_%H%M%S_%f').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
            if start_time <= cs <= end_time and start_time <= ce <= end_time:
                return os.path.join(out, f)
    return None


# 裁剪和绘图
def crop_and_plot(inp, out, q, cs, ce, progress_q):
    try:
        folder_name = os.path.basename(os.path.normpath(inp))
        combined_file = find_combined_file(out, cs, ce, folder_name)
        if not combined_file:
            q.put(('error', '未找到对应时间段的合并文件\n'))
            return
        q.put(('update', f'正在加载 {os.path.basename(combined_file)}...\n'))
        progress_q.put(10)
        data = np.loadtxt(combined_file, dtype=int)
        progress_q.put(30)
        start_time, end_time = parse_combined_filename(os.path.basename(combined_file))
        if not start_time:
            q.put(('error', '无法解析文件名中的时间信息\n'))
            return
        fs = 1000
        q.put(('update', '计算裁剪范围...\n'))
        i0 = int((cs - start_time).total_seconds() * fs)
        i1 = int((ce - start_time).total_seconds() * fs)
        i0, i1 = max(0, i0), min(len(data) - 1, i1)
        if i0 > i1:
            q.put(('error', '指定时间区间无数据\n'))
            return
        q.put(('update', '裁剪数据...\n'))
        seg = data[i0:i1 + 1]
        times = [start_time + timedelta(seconds=i / fs) for i in range(i0, i1 + 1)]
        progress_q.put(50)
        safe_folder_name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', folder_name)
        cropf = os.path.join(out,
                             f'{safe_folder_name}_crop_{cs.strftime("%Y%m%d_%H%M%S")}_{ce.strftime("%Y%m%d_%H%M%S")}.txt')
        np.savetxt(cropf, seg, fmt='%d')
        q.put(('update', f'保存裁剪数据: {cropf}\n'))
        progress_q.put(70)
        q.put(('update', '生成图表...\n'))
        fig = Figure(figsize=(12, 8), dpi=100)
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(times, seg, linewidth=0.8)
        ax1.set_title('时域')
        ax1.grid(True)
        num_ticks = 10
        tick_indices = np.linspace(0, len(times) - 1, num_ticks, dtype=int)
        tick_times = [times[i] for i in tick_indices]
        ax1.set_xticks([times[i] for i in tick_indices])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=ZoneInfo('Asia/Shanghai')))
        fig.autofmt_xdate(rotation=30)
        ax2 = fig.add_subplot(2, 1, 2)
        fft = np.abs(np.fft.rfft(seg))
        freqs = np.linspace(0, fs / 2, len(fft))
        ax2.plot(freqs, fft, linewidth=0.8)
        ax2.set_title('频域')
        ax2.set_xlim(0, 500)
        ax2.grid(True)
        fig.tight_layout()
        canvas = FigureCanvasAgg(fig)
        img = cropf.replace('.txt', '.png')
        canvas.draw()
        canvas.get_renderer()
        canvas.print_figure(img, dpi=300)
        q.put(('update', f'保存图片: {img}\n'))
        q.put(('complete', '裁剪并绘图完成\n'))
    except Exception as ex:
        q.put(('error', f'裁剪出错: {ex}\n'))
    finally:
        progress_q.put(100)


# 解析合并文件名
def parse_combined_filename(filename):
    match = re.match(r'([^_]+)_combined_data_(\d{8}_\d{6}_\d{3})_(\d{8}_\d{6}_\d{3})\.txt', filename)
    if match:
        folder_name, start_str, end_str = match.groups()
        return (
            datetime.strptime(start_str, '%Y%m%d_%H%M%S_%f').replace(tzinfo=ZoneInfo('Asia/Shanghai')),
            datetime.strptime(end_str, '%Y%m%d_%H%M%S_%f').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
        )
    return None, None


class ModernTheme:
    COLORS = {
        'bg': '#f5f5f5',
        'fg': '#333333',
        'highlight': '#4a86e8',
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'button': '#2196f3',
        'button_hover': '#1976d2',
        'frame': '#ffffff',
        'border': '#e0e0e0'
    }
    FONTS = {
        'heading': ('Microsoft YaHei', 11, 'bold'),
        'body': ('Microsoft YaHei', 10),
        'small': ('Microsoft YaHei', 9),
        'mono': ('Consolas', 10)
    }

    @classmethod
    def setup(cls, root):
        style = ttk.Style()
        try:
            if os.name == 'nt':
                style.theme_use('vista')
            else:
                style.theme_use('clam')
        except:
            try:
                style.theme_use('clam')
            except:
                pass
        root.configure(bg=cls.COLORS['bg'])
        style.configure('TLabelframe.Label', foreground=cls.COLORS['fg'], font=cls.FONTS['heading'])
        style.configure('TButton', font=cls.FONTS['body'])
        style.configure('TLabel', foreground=cls.COLORS['fg'], font=cls.FONTS['body'])
        style.configure('TEntry', font=cls.FONTS['body'])
        style.configure('TProgressbar', background=cls.COLORS['highlight'])
        style.configure('Success.TButton', font=cls.FONTS['body'])
        style.configure('Warning.TButton', font=cls.FONTS['body'])
        style.map('Warning.TButton', foreground=[('active', 'white')])
        return style


class CircularProgress(tk.Canvas):
    def __init__(self, parent, size=50, fg='#2196f3', bg='#e0e0e0', width=5, **kwargs):
        canvas_bg = ModernTheme.COLORS['bg']
        super().__init__(parent, width=size, height=size, bg=canvas_bg,
                         highlightthickness=0, **kwargs)
        self.size = size
        self.fg = fg
        self.bg = bg
        self.width = width
        self._angle = 0
        self._running = False
        self.create_arc(self.width, self.width, size - self.width, size - self.width,
                        start=0, extent=359.999, outline=bg, width=width, style='arc')
        self.fg_arc = self.create_arc(self.width, self.width, size - self.width,
                                      size - self.width, start=270, extent=0,
                                      outline=fg, width=width, style='arc')

    def set_progress(self, value):
        angle = value * 3.6
        self.itemconfig(self.fg_arc, extent=angle)

    def start_animation(self):
        self._running = True
        self._animate()

    def stop_animation(self):
        self._running = False

    def _animate(self):
        if not self._running:
            return
        self._angle = (self._angle + 10) % 360
        self.itemconfig(self.fg_arc, start=self._angle, extent=80)
        self.after(50, self._animate)


class BetterScrolledText(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self.config(bg=ModernTheme.COLORS['frame'])
        self.vsb = ttk.Scrollbar(self, orient="vertical")
        self.vsb.pack(side='right', fill='y')
        self.hsb = ttk.Scrollbar(self, orient="horizontal")
        self.hsb.pack(side='bottom', fill='x')
        text_kwargs = {
            'wrap': 'none',
            'font': ModernTheme.FONTS['mono'],
            'background': 'white',
            'foreground': ModernTheme.COLORS['fg'],
            'borderwidth': 1,
            'relief': 'solid',
            'yscrollcommand': self.vsb.set,
            'xscrollcommand': self.hsb.set
        }
        text_kwargs.update(kwargs)
        self.text = tk.Text(self, **text_kwargs)
        self.text.pack(side='left', fill='both', expand=True)
        self.vsb['command'] = self.text.yview
        self.hsb['command'] = self.text.xview
        self.text.tag_configure('info', foreground='#303030')
        self.text.tag_configure('success', foreground=ModernTheme.COLORS['success'])
        self.text.tag_configure('warning', foreground=ModernTheme.COLORS['warning'])
        self.text.tag_configure('error', foreground=ModernTheme.COLORS['error'])
        self.text.tag_configure('highlight', foreground=ModernTheme.COLORS['highlight'])

    def insert(self, index, text, tags=None):
        self.text.insert(index, text, tags)

    def see(self, index):
        self.text.see(index)

    def clear(self):
        self.text.delete(1.0, tk.END)


class DatCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title('.dat文件处理工具')
        self.root.geometry('900x700')
        self.root.minsize(800, 600)
        self.theme = ModernTheme.setup(root)
        self.input_folder = tk.StringVar()
        self.start_str = tk.StringVar()
        self.end_str = tk.StringVar()
        self.merged_intervals = []
        self.queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.create_widgets()
        self.processing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        top_frame = ttk.LabelFrame(main_frame, text='数据文件设置', padding=10)
        top_frame.pack(fill=tk.X, pady=5)

        input_frame = ttk.Frame(top_frame)
        input_frame.pack(fill=tk.X, pady=5)
        ttk.Label(input_frame, text='输入文件夹:', width=10).pack(side=tk.LEFT)
        ttk.Entry(input_frame, textvariable=self.input_folder).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text='浏览...', command=self.browse_input).pack(side=tk.RIGHT)

        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text='分析时间段', command=self.preprocess, style='Success.TButton').pack(fill=tk.X,
                                                                                                        pady=5)
        ttk.Button(btn_frame, text='合并数据', command=self.combine).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(btn_frame, text='裁剪并生成图像', command=self.crop).pack(side=tk.RIGHT, expand=True, fill=tk.X,
                                                                             padx=5)

        time_frame = ttk.LabelFrame(main_frame, text='裁剪时间区间设置', padding=10)
        time_frame.pack(fill=tk.X, pady=5)
        time_input_frame = ttk.Frame(time_frame)
        time_input_frame.pack(fill=tk.X)
        ttk.Label(time_input_frame, text='开始时间:').pack(side=tk.LEFT)
        ttk.Entry(time_input_frame, textvariable=self.start_str, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_input_frame, text='结束时间:').pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(time_input_frame, textvariable=self.end_str, width=25).pack(side=tk.LEFT, padx=5)
        # 更新格式提示
        ttk.Label(time_input_frame, text='格式: YYYYMMDD_HHMMSS').pack(side=tk.RIGHT)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.spinner = CircularProgress(status_frame, size=24, fg=ModernTheme.COLORS['highlight'])
        self.spinner.pack(side=tk.RIGHT, padx=5)
        self.status_var = tk.StringVar(value='就绪')
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, width=10)
        self.status_label.pack(side=tk.RIGHT)

        log_frame = ttk.LabelFrame(main_frame, text='处理日志', padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log = BetterScrolledText(log_frame)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.log.insert(tk.END, ".dat文件处理工具\n", 'highlight')
        self.log.insert(tk.END, "使用说明:\n", 'info')
        self.log.insert(tk.END, "1. 选择输入文件夹\n", 'info')
        self.log.insert(tk.END, "2. 点击'分析时间段'查看可用数据范围\n", 'info')
        self.log.insert(tk.END, "3. 点击'合并数据'将所有.dat文件合并\n", 'info')
        self.log.insert(tk.END, "4. 设置时间区间（格式：YYYYMMDD_HHMMSS），然后点击'裁剪并生成图像'\n", 'info')
        self.log.insert(tk.END, "5. 所有输出文件将保存在输入文件夹的output子目录中\n", 'info')
        self.log.insert(tk.END, "就绪，等待选择文件夹...\n", 'success')

        footer = ttk.Frame(self.root, relief=tk.SUNKEN, padding=5)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(footer, text="© 2025 .dat处理工具", font=ModernTheme.FONTS['small']).pack(side=tk.LEFT)
        cpu_count = mp.cpu_count()
        ttk.Label(footer, text=f"系统CPU核心: {cpu_count}", font=ModernTheme.FONTS['small']).pack(side=tk.RIGHT)

    # 裁剪
    def crop(self):
        inp = self.input_folder.get()
        out = os.path.join(inp, 'output')
        os.makedirs(out, exist_ok=True)

        s, e = self.start_str.get(), self.end_str.get()
        if not inp or not s or not e:
            messagebox.showerror('错误', '请选择输入文件夹并填写时间区间')
            return
        if self.processing:
            messagebox.showinfo('处理中', '请等待当前操作完成')
            return
        try:
            # 毫秒默认为0
            try:
                cs = datetime.strptime(s, '%Y%m%d_%H%M%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
            except ValueError:
                messagebox.showerror('错误', f'开始时间格式错误，请使用YYYYMMDD_HHMMSS格式')
                return

            try:
                ce = datetime.strptime(e, '%Y%m%d_%H%M%S').replace(tzinfo=ZoneInfo('Asia/Shanghai'))
            except ValueError:
                messagebox.showerror('错误', f'结束时间格式错误，请使用YYYYMMDD_HHMMSS格式')
                return
        except Exception as ex:
            messagebox.showerror('错误', f'时间格式错误\n错误详情: {ex}')
            return
        self.set_processing_state(True)
        self.status_var.set('裁剪中...')
        self.log.insert(tk.END, f"开始裁剪数据 [{s} 到 {e}]...\n", 'highlight')
        self.log.insert(tk.END, f"输出文件夹: {out}\n", 'info')
        threading.Thread(target=crop_and_plot,
                         args=(inp, out, self.queue, cs, ce, self.progress_queue),
                         daemon=True).start()
        self.root.after(100, self.check_queue)

    def browse_input(self):
        directory = filedialog.askdirectory(title='选择输入文件夹')
        if directory:
            self.input_folder.set(directory)
            self.log.insert(tk.END, f"已选择输入文件夹: {directory}\n", 'info')

    def preprocess(self):
        inp = self.input_folder.get()
        if not inp:
            messagebox.showerror('错误', '请选择输入文件夹')
            return
        out = os.path.join(inp, 'output')
        os.makedirs(out, exist_ok=True)

        self.log.insert(tk.END, "开始分析时间段...\n", 'highlight')
        self.log.insert(tk.END, f"输出文件夹: {out}\n", 'info')
        all_files = sorted(f for f in os.listdir(inp) if f.endswith('.dat'))
        if not all_files:
            self.log.insert(tk.END, "错误: 输入文件夹中没有找到.dat文件\n", 'error')
            return
        self.log.insert(tk.END, f"找到 {len(all_files)} 个.dat文件，分析中...\n", 'info')
        self.set_processing_state(True)
        self.status_var.set('分析中...')
        threading.Thread(target=self._preprocess_thread, args=(inp, all_files), daemon=True).start()
        self.root.after(100, self.check_queue)

    def _preprocess_thread(self, inp, all_files):
        try:
            intervals = []
            total = len(all_files)
            for i, fname in enumerate(all_files):
                if i % 10 == 0:
                    progress = int((i / total) * 100)
                    self.progress_queue.put(progress)
                try:
                    file_path = os.path.join(inp, fname)
                    st = parse_filename_to_time(fname)
                    file_size = os.path.getsize(file_path)
                    data_len = file_size // 4
                    ed = st + timedelta(seconds=(data_len - 1) / 1000)
                    intervals.append((st, ed))
                except Exception as ex:
                    self.queue.put(('update', f"解析 {fname} 错误: {ex}\n", 'error'))
            merged = merge_intervals(intervals)
            self.merged_intervals = merged
            self.queue.put(('update', '\n可用时间段:\n', 'highlight'))
            for s, e in merged:
                time_str = f"{s.strftime('%Y%m%d_%H%M%S_%f')[:-3]} - {e.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
                self.queue.put(('update', time_str + '\n', 'success'))
            self.queue.put(('complete', '时间段分析完成\n'))
        except Exception as ex:
            self.queue.put(('error', f'分析时间段时出错: {ex}\n'))
        finally:
            self.progress_queue.put(100)

    def set_processing_state(self, is_processing):
        self.processing = is_processing
        if is_processing:
            self.progress['value'] = 0
            self.spinner.start_animation()
        else:
            self.progress['value'] = 100
            self.spinner.stop_animation()
            self.status_var.set('就绪')

    def check_queue(self):
        try:
            while True:
                progress = self.progress_queue.get_nowait()
                self.progress['value'] = progress
                self.progress_queue.task_done()
        except queue.Empty:
            pass
        try:
            while True:
                message = self.queue.get_nowait()
                message_type = message[0]
                if message_type == 'update':
                    text = message[1]
                    tag = message[2] if len(message) > 2 else None
                    self.log.insert(tk.END, text, tag)
                    self.log.see(tk.END)
                elif message_type == 'error':
                    error_msg = message[1]
                    self.log.insert(tk.END, f"错误: {error_msg}\n", 'error')
                    self.log.see(tk.END)
                    messagebox.showerror('错误', error_msg)
                    self.set_processing_state(False)
                elif message_type == 'complete':
                    complete_msg = message[1]
                    self.log.insert(tk.END, f"{complete_msg}\n", 'success')
                    self.log.see(tk.END)
                    messagebox.showinfo('完成', complete_msg)
                    self.set_processing_state(False)
                self.queue.task_done()
        except queue.Empty:
            if self.processing:
                self.root.after(100, self.check_queue)
            return

    def combine(self):
        inp = self.input_folder.get()
        if not inp:
            messagebox.showerror('错误', '请选择输入文件夹')
            return
        out = os.path.join(inp, 'output')
        os.makedirs(out, exist_ok=True)

        if self.processing:
            messagebox.showinfo('处理中', '请等待当前操作完成')
            return
        self.set_processing_state(True)
        self.status_var.set('合并中...')
        self.log.insert(tk.END, "开始合并数据文件...\n", 'highlight')
        self.log.insert(tk.END, f"输出文件夹: {out}\n", 'info')
        threading.Thread(target=combine_data,
                         args=(inp, out, self.queue, self.progress_queue, self.merged_intervals),
                         daemon=True).start()
        self.root.after(100, self.check_queue)

    def on_closing(self):
        if self.processing:
            if messagebox.askokcancel("确认", "处理仍在进行中，确定要退出吗？"):
                self.root.destroy()
        else:
            self.root.destroy()


if __name__ == '__main__':
    import ctypes

    try:
        p = ctypes.cdll.kernel32.GetCurrentProcess()
        ctypes.cdll.kernel32.SetPriorityClass(p, 0x00000080)
    except:
        pass
    mp.set_start_method('spawn', force=True)
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 1.2)
    app = DatCropApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
