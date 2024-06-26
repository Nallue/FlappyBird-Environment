import pygame
from pygame.locals import *
import random
import time
import math
import numpy as np

from deep_q_learning import DQNAgent


state_size = 7 
action_size = 2 # saltar o no saltar
agent = DQNAgent(state_size, action_size, epsilon_decay=0.999, memory_size=1000, batch_size=100, epsilon_min=0.0005)
cont_muertes = 0

pygame.init()

clock = pygame.time.Clock()
fps = 60

screen_width = 864
screen_height = 936

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

#define font
font = pygame.font.SysFont('Bauhaus 93', 60)
font_info = pygame.font.SysFont('System Negrita', 30)

#define colours
white = (255, 255, 255)
red = (255, 0, 0)
black = (0, 0, 0)

#define game variables
ground_scroll = 0
scroll_speed = 4
flying = False
game_over = False
pipe_gap = 150
pipe_frequency = 1500 #milliseconds
last_pipe = pygame.time.get_ticks() - pipe_frequency
score = 0
last_score = 0
pass_pipe = False


#load images
bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')


#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))

def reset_game():
	pipe_group.empty()
	flappy.rect.x = 100
	flappy.rect.y = int(screen_height / 2)
	score = 0
	return score


class Bird(pygame.sprite.Sprite):

	def __init__(self, x, y):
		pygame.sprite.Sprite.__init__(self)
		self.images = []
		self.index = 0
		self.counter = 0
		for num in range (1, 4):
			img = pygame.image.load(f"img/bird{num}.png")
			self.images.append(img)
		self.image = self.images[self.index]
		self.rect = self.image.get_rect()
		self.rect.center = [x, y]
		self.vel = 0
		self.clicked = False

	def update(self, action, bird_group, pipe_group):

		global game_over
		global flying
		if game_over == False:
			#jump
			if action == 1: #and self.clicked == False:
				self.clicked = True
				self.vel = -10
			else:
				self.clicked = False
			
			if flying == True:
				#apply gravity
				self.vel += 0.5
				if self.vel > 8:
					self.vel = 8
				if self.rect.bottom < 768:
					self.rect.y += int(self.vel)

			#handle the animation
			flap_cooldown = 5
			self.counter += 1
			
			if self.counter > flap_cooldown:
				self.counter = 0
				self.index += 1
				if self.index >= len(self.images):
					self.index = 0
				self.image = self.images[self.index]


			#rotate the bird
			self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
		else:
			#point the bird at the ground
			self.image = pygame.transform.rotate(self.images[self.index], -90)

		reward = 0.01
		done = False
		#look for collision
		if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0 or flappy.rect.bottom >= 768:
			game_over = True
			flying = False

			reward = -20
			done = True
		
		else:
			global last_score
			if score > last_score:
				reward += 10
		
		return (reward, done)


		
	# def update(self, action):

	# 	if game_over == False:
	# 		#jump with mouse
	# 		# if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
	# 		# 	self.clicked = True
	# 		# 	self.vel = -10
	# 		# if pygame.mouse.get_pressed()[0] == 0:
	# 		# 	self.clicked = False


	# 		#agent learn
	# 		agent.learn(last_state, last_action, 0.01, state, False)

	# 		# agent jump
	# 		last_action = agent.act(np.reshape(state, [1, state_size]))
	# 		if last_action == 1: #and self.clicked == False:
	# 			self.clicked = True
	# 			self.vel = -10
	# 		else:
	# 			self.clicked = False

	# 		if flying == True:
	# 			#apply gravity
	# 			self.vel += 0.5
	# 			if self.vel > 8:
	# 				self.vel = 8
	# 			if self.rect.bottom < 768:
	# 				self.rect.y += int(self.vel)

	# 		last_state = state



	# 		#handle the animation
	# 		flap_cooldown = 5
	# 		self.counter += 1
			
	# 		if self.counter > flap_cooldown:
	# 			self.counter = 0
	# 			self.index += 1
	# 			if self.index >= len(self.images):
	# 				self.index = 0
	# 			self.image = self.images[self.index]


	# 		#rotate the bird
	# 		self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
	# 	else:

	# 		reward = -100
	# 		done = True
	# 		print(np.reshape(last_state, [1, state_size]), last_action, reward, np.reshape(state, [1, state_size]), done)
	# 		agent.learn(last_state, last_action, reward, state, done)

	# 		#point the bird at the ground
	# 		self.image = pygame.transform.rotate(self.images[self.index], -90)
	# 		self.vel = 0



class Pipe(pygame.sprite.Sprite):

	def __init__(self, x, y, position):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.image.load("img/pipe.png")
		self.rect = self.image.get_rect()
		#position variable determines if the pipe is coming from the bottom or top
		#position 1 is from the top, -1 is from the bottom
		if position == 1:
			self.image = pygame.transform.flip(self.image, False, True)
			self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
		elif position == -1:
			self.rect.topleft = [x, y + int(pipe_gap / 2)]


	def update(self):
		self.rect.x -= scroll_speed
		if self.rect.right < 0:
			self.kill()



class Button():
	def __init__(self, x, y, image):
		self.image = image
		self.rect = self.image.get_rect()
		self.rect.topleft = (x, y)

	def draw(self):
		action = False

		#get mouse position
		pos = pygame.mouse.get_pos()

		#check mouseover and clicked conditions
		if self.rect.collidepoint(pos):
			if pygame.mouse.get_pressed()[0] == 1:
				action = True

		#draw button
		screen.blit(self.image, (self.rect.x, self.rect.y))

		return action


def lanzar_rayo_horizontal_derecha(sprite1, sprite_group):
    start_pos = sprite1.rect.centery
    min_distance = screen_width  # Empezamos con la distancia máxima posible (ancho de la pantalla)
    closest_sprite = None
    collision_point = None

    for sprite in sprite_group:
        min_y = sprite.rect.top
        max_y = sprite.rect.bottom 
        if min_y <= start_pos <= max_y and sprite.rect.left > sprite1.rect.right:
            distance = sprite.rect.left - sprite1.rect.right
            if distance < min_distance:
                min_distance = distance
                closest_sprite = sprite
                collision_point = (sprite.rect.left, start_pos)

    if closest_sprite:
        # Dibujar el rayo en la pantalla
        pygame.draw.line(screen, red, (sprite1.rect.right, sprite1.rect.centery), collision_point, 2)
    
    return min_distance if closest_sprite else -1

def lanzar_rayo_altura(sprite1, base):
    start_pos = (sprite1.rect.centerx, sprite1.rect.bottom)
    collision_point = (sprite1.rect.centerx, base)

    if (base-sprite1.rect.bottom)>0:
        pygame.draw.line(screen, red, start_pos, collision_point, 2)
    
    return (base-sprite1.rect.bottom) if (base-sprite1.rect.bottom)>0 else -1

def lanzar_rayo_diagonal(sprite1, sprite_group, ang):
    start_pos = (sprite1.rect.right, sprite1.rect.centery)
    min_distance = float('inf')
    closest_sprite = None
    collision_point = None

    # Calculamos el ángulo en radianes para una diagonal de 45 grados
    angle = math.radians(ang)
    dx = math.cos(angle)
    dy = -math.sin(angle)

    # Recorrer una distancia máxima igual a la diagonal de la pantalla
    for i in range(int(math.hypot(screen_width, screen_height))):
        x = start_pos[0] + i * dx
        y = start_pos[1] + i * dy

        # Crear un punto de prueba
        point = (x, y)

        for sprite in sprite_group:
            if sprite.rect.collidepoint(point):
                distance = math.hypot(x - start_pos[0], y - start_pos[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_sprite = sprite
                    collision_point = point

    if closest_sprite:
        # Dibujar el rayo en la pantalla
        pygame.draw.line(screen, red, start_pos, collision_point, 2)
    
    return min_distance if closest_sprite else -1


pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

flappy = Bird(100, int(screen_height / 2))

bird_group.add(flappy)

#create restart button instance
button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)

def get_state():
	state = []
	dist_horizontal = (lanzar_rayo_horizontal_derecha(flappy, pipe_group))
	state.append(dist_horizontal)
	
	dist_diag_TR1 = (lanzar_rayo_diagonal(flappy, pipe_group, 45))
	state.append(dist_diag_TR1)
	
	dist_diag_BR1 = (lanzar_rayo_diagonal(flappy, pipe_group, -45))
	state.append(dist_diag_BR1)

	dist_diag_TR2 = (lanzar_rayo_diagonal(flappy, pipe_group, 22.5))
	state.append(dist_diag_TR2)

	dist_diag_BR2 = (lanzar_rayo_diagonal(flappy, pipe_group, -22.5))
	state.append(dist_diag_BR2)

	dist_suelo = (lanzar_rayo_altura(flappy, bg.get_height()))
	state.append(dist_suelo)
	state.append(flappy.vel) #Añadir velocidad al estado
	return state

run = True
while run:

	clock.tick(fps)

	#draw background
	screen.blit(bg, (0,0))

	pipe_group.draw(screen)
	bird_group.draw(screen)

	#draw and scroll the ground
	screen.blit(ground_img, (ground_scroll, 768))
	
    #estado
	state = get_state()
	action = agent.act(state)
	reward, done = flappy.update(action, bird_group, pipe_group)
	new_state = get_state()
	#learn(self, state, action, reward, next_state, done):
	agent.store_transition(state, action, reward, new_state, done)
	agent.learn()

	#check the score
	if len(pipe_group) > 0:
		if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left\
			and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right\
			and pass_pipe == False:
			pass_pipe = True
		if pass_pipe == True:
			if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
				score += 1
				pass_pipe = False
	draw_text(str(score), font, white, int(screen_width / 2), 20)




	if flying == True and game_over == False:
		#generate new pipes
		time_now = pygame.time.get_ticks()
		if time_now - last_pipe > pipe_frequency:
			pipe_height = random.randint(-100, 100)
			btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
			top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
			pipe_group.add(btm_pipe)
			pipe_group.add(top_pipe)
			last_pipe = time_now

		pipe_group.update()

		ground_scroll -= scroll_speed
		if abs(ground_scroll) > 35:
			ground_scroll = 0
	

	if((cont_muertes%100)==0 and cont_muertes!=0):
		agent.set_train_off()
		epsi_img1 = font_info.render("Train OFF", True, red)
		screen.blit(epsi_img1, (int(screen_width -250), int(50)))
		agent.save_model("./NN")
	else:
		agent.set_train_on()
		epsi_img2 = font_info.render("Train On", True, white)
		screen.blit(epsi_img2, (int(screen_width -250), int(50)))
	
	muertes_txt = font_info.render("Muertes: "+str(cont_muertes), True, black)
	screen.blit(muertes_txt, (int(150), int(50)))

	#check for game over and reset
	if game_over == True:
		cont_muertes += 1
		print("MUERTES:",cont_muertes)
		if True or button.draw():
			game_over = False
			score = reset_game()
			flying = True
			time.sleep(0.2)


	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if event.type == pygame.MOUSEBUTTONDOWN and flying == False and game_over == False:
			flying = True

	pygame.display.update()

pygame.quit()