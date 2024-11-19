"""
Interate f = z**2 + c
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tqdm
from scipy.optimize import fminbound


class AnimateJulia:

    def __init__(self):
        self.maxiter = 100
        self.shape = (480, 720)
        self.shape = (960, 1440)
        self.num_frame = 256
        self.interval = 30
        self.up_down = False

        self.cache = np.zeros((self.num_frame, ) + self.shape, dtype=np.uint16)
        self.ctr = 0
        self.count_up = True
        self.construction_complete = False

        self.fig, self.ax = plt.subplots()
        self.extents = (-1.5, 1.5, -1, 1)
        self.ax.set_xlim(self.extents[:2])
        self.ax.set_ylim(self.extents[2:])
        self.ax.axis(False)

        self.pbar = tqdm.tqdm(total=self.num_frame)

        img = calc_julia(self.shape, self.pos(0), extents=self.extents, maxiter=self.maxiter)
        self._ax_img = self.ax.imshow(img, extent=self.extents)
        self._ax_img.set_clim((0, self.maxiter - 1))

        ani = FuncAnimation(
            self.fig, self.update,
            frames=np.linspace(0, 
                               2 * np.pi if self.up_down else np.pi,
                               self.num_frame),
            blit=True, interval=self.interval)
        
        plt.show()

    def pos(self, phi):
        # return self.left_circle(phi)  
        return self.upper_rt_lobe(phi)

    def upper_rt_lobe(self, phi):
        xy = subcircle(1, 4, phi)
        return xy[0] + 1j * xy[1]
        
    def left_circle(self, phi):
        r = 0.25 * 1.02
        x = r * np.cos(phi) - 1
        y = r * np.sin(phi)
        return x + 1j * y

    def cardiod(self, phi):
        a = 0.25 * 1.02
        x = 2 * a * (1 - np.cos(phi)) * np.cos(phi) + 0.25
        y = 2 * a * (1 - np.cos(phi)) * np.sin(phi)
        return x + 1j * y

    def update(self, frame):
        # print(frame / (2 * np.pi))
        if not self.construction_complete:
            self.pbar.update()
            img = calc_julia(self.shape, self.pos(frame), extents=self.extents,
                            maxiter=self.maxiter)
            self.cache[self.ctr] = img
            self.ctr += 1

            if self.ctr == self.num_frame:
                self.construction_complete = True
                if self.up_down:
                    self.ctr -= 1
                    self.count_up = False
                else:
                    self.ctr = 0
        else:
            img = self.cache[self.ctr]
            if self.count_up:
                self.ctr += 1
            else:
                self.ctr -= 1

            if self.ctr == self.num_frame:
                if self.up_down:
                    self.ctr -= 1
                    self.count_up = False
                else:
                    self.ctr = 0
            elif self.ctr == -1:
                self.ctr = 0
                self.count_up = True

        # print("img.max()", img.max())
        self._ax_img.set_array(img)
        # self._ax_img.set_clim((img.min(), img.max()))
        return self._ax_img,


def calc_mandelbrot(shape, extents=(-2, 0.5, -1.2, 1.2), maxiter=50, threshold=2):
    xmin, xmax, ymin, ymax = extents
    img = np.zeros(shape, dtype=np.uint16)
    val = np.zeros(shape)
    re, im = np.meshgrid(np.linspace(xmin, xmax, shape[1]),
                         np.linspace(ymin, ymax, shape[0]))

    for n in range(maxiter):
        val = val**2 + re + im * 1j
        img[(img == 0) & (np.abs(val) >= threshold)] = n

    # return np.abs(val), img
    return img


def calc_julia(shape, c, maxiter=50, threshold=2, extents=(-1.5, 1.5, -1, 1)):
    """
    c is held constant
    """
    xmin, xmax, ymin, ymax = extents

    img = np.zeros(shape, dtype=np.uint16)
    re, im = np.meshgrid(np.linspace(xmin, xmax, shape[1]),
                         np.linspace(ymin, ymax, shape[0]))
    
    val = re + 1j * im

    for n in range(maxiter):
        val = val**2 + c
        img[(img == 0) & (np.abs(val) >= threshold)] = n

    # return np.abs(val), img
    return img


class ZoomableFigure:

    def __init__(self, func, xmin, xmax, ymin, ymax):
        self._fig, self.ax = plt.subplots()
        self.ax.set_xlim((xmin, xmax))
        self.ax.set_ylim((ymin, ymax))
        # self.ax.axis('equal')
        self.ax.axis(False)
        self._func = func

        img = self._func(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        self._ax_img = self.ax.imshow(img, extent=(xmin, xmax, ymin, ymax))
        print("clim", self._ax_img.get_clim())
        # self._fig.canvas.mpl_connect('xlim_changed', self.onevent)
        self._xlim_event_triggered = False
        self._ylim_event_triggered = False
        self._xcallback = self.ax.callbacks.connect('xlim_changed', self.xlim_changed)
        self._ycallback = self.ax.callbacks.connect('ylim_changed', self.ylim_changed)

    def xlim_changed(self, event):
        self._xlim_event_triggered = True
        if self._xlim_event_triggered and self._ylim_event_triggered:
            self.onevent()

    def ylim_changed(self, event):
        self._ylim_event_triggered = True
        if self._xlim_event_triggered and self._ylim_event_triggered:
            self.onevent()

    def onevent(self):
        # print("event", event)
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        print("extents", xmin, xmax, ymin, ymax)
        print("ax imgage extents:", self._ax_img.get_extent())

        img = self._func(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        # self.ax.imshow(img, extent=(xmin, xmax, ymin, ymax))
        self._ax_img.set_array(img[::-1, :])
        self.ax.callbacks.disconnect(self._xcallback)
        self.ax.callbacks.disconnect(self._ycallback)
        print("new clim", (img.min(), img.max()))
        # self._ax_img.set_clim((img.min(), img.max()))
        self._ax_img.set_extent((xmin, xmax, ymin, ymax))
        
        self._fig.canvas.draw()

        self._xcallback = self.ax.callbacks.connect('xlim_changed', self.xlim_changed)
        self._ycallback = self.ax.callbacks.connect('ylim_changed', self.ylim_changed)

        self._xlim_event_triggered = False
        self._ylim_event_triggered = False


def cardiod(phi):
    a = 0.25
    x = 2 * a * (1 - np.cos(phi)) * np.cos(phi) + 0.25
    y = 2 * a * (1 - np.cos(phi)) * np.sin(phi)
    # return np.array([x, y])
    return x + 1j * y


def cpq(p, q):
    """
    Wikipedia article
    """
    r = np.exp(2 * np.pi * 1j * p / q)
    return r / 2 * (1 - r / 2)


def cpq_to_phi(p, q):
    pos = cpq(p, q)
    def err(phi):
        return np.abs(pos - cardiod(phi))**2
    return fminbound(err, 0, 2 * np.pi)


def radius(p, q):
    """
    https://www.sciencedirect.com/science/article/pii/S259005441930017X
    """
    return 1 / q**2 * np.sin(np.pi * p / q)


def cardiod_normal(phi):
    a = 0.25
    dx = 2 * a * (-np.sin(phi) + np.sin(2 * phi))
    dy = 2 * a * (np.sin(phi)**2 + np.cos(phi) * (1 - np.cos(phi)))
    ret = np.array([dy, -dx])
    ret /= np.linalg.norm(ret)
    return ret


def subcircle(p, q, phi):
    r = radius(p, q)
    cardiod_pos = cpq(p, q)
    cardiod_pos = np.array([np.real(cardiod_pos), np.imag(cardiod_pos)])

    phi_cardiod = cpq_to_phi(p, q)
    
    n = cardiod_normal(phi_cardiod)

    center = cardiod_pos + n * r

    return np.vstack([
        center[0] + r * np.cos(phi),
        center[1] + r * np.sin(phi),
    ])


def plot_lobes():
    extents=(-2, 0.5, -1.2, 1.2)

    img = calc_mandelbrot((480, 270), extents)

    plt.imshow(img, extent=extents)

    phi = np.linspace(0, 2 * np.pi, 100)

    card = cardiod(phi)
    plt.plot(np.real(card), np.imag(card))
    # plt.plot(*subcircle(1, 2, phi))
    # plt.plot(*subcircle(1, 3, phi))
    plt.plot(*subcircle(1, 4, phi))



if __name__ == '__main__':
    pass

    # plt.ion()

    # c = -0.743517833 + 1j * -0.127094578
    # func = lambda xmin, xmax, ymin, ymax: calc_julia((480, 760), c, 
    #                                                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
    #                                                  maxiter=200)
    # f = ZoomableFigure(func, 
    #                    -1.5, 1.5, -1, 1)

    # c = -0.743517833 + 1j * -0.127094578
    # func = lambda xmin, xmax, ymin, ymax: calc_mandelbrot((480, 760),  
    #                                                  xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
    #                                                  maxiter=50)
    # f = ZoomableFigure(func, 
    #                    -1.5, .5, -1, 1)

    # phi = np.linspace(0, 2* np.pi, 100)
    # a = 0.25 * 1.01
    # x = 2 * a * (1 - np.cos(phi)) * np.cos(phi) + 0.25
    # y = 2 * a * (1 - np.cos(phi)) * np.sin(phi)

    # plt.plot(x, y)

    # r = 0.7885
    # x = r*np.cos(phi)
    # y = r*np.sin(phi)
    # plt.plot(x, y)
    # plt.axis('equal')

    a = AnimateJulia()


    # plt.ion()

    # xmin, xmax, ymin, ymax = (-2, .5, -1, 1)
    # mval, mimg = calc_mandelbrot((480, 760), xmin, xmax, ymin, ymax, 50)

    # plt.figure(1)
    # plt.imshow(mimg, extent=(xmin, xmax, ymin, ymax))
    # plt.axis('equal')
    # plt.axis(False)

    # c = -0.743517833 + 1j * -0.127094578
    # xmin=-1.5
    # xmax=1.5
    # ymin=-1
    # ymax=1
    # jval, jimg = calc_julia((480, 760), c, maxiter=150, threshold=2)

    # plt.figure(2)
    # plt.imshow(jimg, extent=(xmin, xmax, ymin, ymax))
    # plt.axis('equal')
    # plt.axis(False)

        
