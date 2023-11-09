import pygame
from squaregrid import SquareGrid

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400

colormap = {
    -1: (0,0,255),
    0: (0, 255, 0),
    1: (255, 255, 0),
    2: (255, 165, 0),
    3: (255, 0, 0)
}

def main(grid: SquareGrid):
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    while True:
        drawGrid(grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()


def drawGrid(grid: SquareGrid):
    blockSize = int(WINDOW_WIDTH / grid.n)

    for i_r, row in enumerate(grid.grid):
        for i_c, cell in enumerate(row):
            x = i_c * blockSize
            y = i_r * blockSize
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.Surface.fill(SCREEN, colormap[cell], rect)

grid = SquareGrid([[0,-1,0,1],
                   [1,2,1,0],
                   [2,3,2,0],
                   [1,2,1,0]])

main(grid)