import pygame
from pygame.locals import *
import random
import time
import math


class FlappyBird():
	def __init__(self, show_window=1):
		pygame.init()
		
		#render mode
			#render mode=0
		if show_window==0:
			self.show_window=False
		else:
			self.show_window=True

		#screen size
		self.screen_width = 864
		self.screen_height = 936
		
		#define font
		self.font = pygame.font.SysFont('Bauhaus 93', 60)
		self.font_info = pygame.font.SysFont('System Negrita', 30)

		#colors
		self.white = (255, 255, 255)
		self.red = (255, 0, 0)
		self.black = (0, 0, 0)

		#images
		self.bg = pygame.image.load('img/bg.png')
		self.ground_img = pygame.image.load('img/ground.png')
		self.button_img = pygame.image.load('img/restart.png')
		self.pipe_img = pygame.image.load("img/pipe.png")

		#game properties
		self.pipe_gap = 160
		self.pipe_frequency = 80 #times action has been called

	class Bird(pygame.sprite.Sprite):

		def __init__(self, x, y, clase_externa):
			self.clase_externa = clase_externa
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

		def update(self, act):

			if self.clase_externa.flying == True:
				#apply gravity
				self.vel += 0.5
				if self.vel > 8:
					self.vel = 8
				if self.rect.bottom < 768:
					self.rect.y += int(self.vel)

			if self.clase_externa.game_over == False:
				#jump
				if act == 1 and self.clicked == False:
					self.clicked = True
					self.vel = -10
				if act == 0:
					self.clicked = False

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

	class Pipe(pygame.sprite.Sprite):

		def __init__(self, x, y, position, clase_externa):
			self.clase_externa = clase_externa
			pygame.sprite.Sprite.__init__(self)
			self.image = clase_externa.pipe_img
			self.rect = self.image.get_rect()
			#position variable determines if the pipe is coming from the bottom or top
			#position 1 is from the top, -1 is from the bottom
			self.position = position
			if position == 1:
				self.image = pygame.transform.flip(self.image, False, True)
				self.rect.bottomleft = [x, y - int(self.clase_externa.pipe_gap / 2)]
			elif position == -1:
				self.rect.topleft = [x, y + int(self.clase_externa.pipe_gap / 2)]


		def update(self):
			self.rect.x -= self.clase_externa.scroll_speed
			if self.rect.right < 0:
				self.kill()

	def get_state(self):
		def lanzar_rayo_horizontal_distancia_tubo(sprite1, sprite_group, screen, screen_width, color):
			start_pos = sprite1.rect.centery
			min_distance = screen_width+self.pipe_img.get_width()  # Empezamos con la distancia máxima posible (ancho de la pantalla + pipe)
			closest_sprite = None

			for sprite in sprite_group:
				if sprite.rect.right > sprite1.rect.left and sprite.position == -1:
					distance = sprite.rect.right - sprite1.rect.left
					if distance <= min_distance:
						min_distance = distance
						closest_sprite = sprite
						collision_point = (sprite.rect.right, start_pos)

			if closest_sprite:
				# Dibujar el rayo en la pantalla
				pygame.draw.line(screen, color, (sprite1.rect.left, sprite1.rect.bottom), (sprite1.rect.left, closest_sprite.rect.top), 2)
				pygame.draw.line(screen, color, (sprite1.rect.right, sprite1.rect.top), (sprite1.rect.right, closest_sprite.rect.top-self.pipe_gap), 2)

			return_list = [closest_sprite.rect.top - sprite1.rect.bottom, closest_sprite.rect.top-self.pipe_gap - sprite1.rect.top, self.flappy.vel]

			rwd = 0
			if sprite1.rect.top > closest_sprite.rect.top-150 and sprite1.rect.bottom < closest_sprite.rect.top:
				rwd = 0.5
			
			return return_list, rwd

		

		# def lanzar_rayo_diagonal(sprite1, sprite_group, ang, screen_width, screen_height, screen, color):
		# 	start_pos = (sprite1.rect.right, sprite1.rect.centery)
		# 	min_distance = float('inf')
		# 	closest_sprite = None
		# 	collision_point = None

		# 	# Calculamos el ángulo en radianes para una diagonal de 45 grados
		# 	angle = math.radians(ang)
		# 	dx = math.cos(angle)
		# 	dy = -math.sin(angle)

		# 	# Recorrer una distancia máxima igual a la diagonal de la pantalla
		# 	for i in range(int(math.hypot(screen_width, screen_height))):
		# 		x = start_pos[0] + i * dx
		# 		y = start_pos[1] + i * dy

		# 		# Crear un punto de prueba
		# 		point = (x, y)

		# 		for sprite in sprite_group:
		# 			if sprite.rect.collidepoint(point):
		# 				distance = math.hypot(x - start_pos[0], y - start_pos[1])
		# 				if distance < min_distance:
		# 					min_distance = distance
		# 					closest_sprite = sprite
		# 					collision_point = point

		# 	if closest_sprite:
		# 		# Dibujar el rayo en la pantalla
		# 		pygame.draw.line(screen, color, start_pos, collision_point, 2)
			
		# 	return min_distance if closest_sprite else -1

		
		state, rwd = (lanzar_rayo_horizontal_distancia_tubo(self.flappy, self.pipe_group, self.screen, self.screen_width, self.red))
		
		return state, rwd

	def draw_text(self, text, font, text_col, x, y):
		img = font.render(text, True, text_col)
		self.screen.blit(img, (x, y))

	def reset(self):
		#varaibles de juego
		self.flying = True
		self.game_over = False

		self.score = 0
		self.pass_pipe = False

		self.cont_pipe_timer = 0

		self.ground_scroll = 0
		self.scroll_speed = 4

		#create window
		if(self.show_window):
			self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
			pygame.display.set_caption('Flappy Bird')
		else:
			self.screen = pygame.Surface((self.screen_width, self.screen_height))

		#create sprite Groups
		self.pipe_group = pygame.sprite.Group()
		self.bird_group = pygame.sprite.Group()

		#generate bird
		self.flappy = self.Bird(100, int(self.screen_height / 2), self)
		self.bird_group.add(self.flappy)

		#generate first pipe
			# self.cont_pipe_timer = 0 already done before
		pipe_height = random.randint(-100, 100)
		btm_pipe = self.Pipe(self.screen_width, int(self.screen_height / 2) + pipe_height, -1, self)
		top_pipe = self.Pipe(self.screen_width, int(self.screen_height / 2) + pipe_height, 1, self)
		self.pipe_group.add(btm_pipe)
		self.pipe_group.add(top_pipe)

		#draw background
		self.screen.blit(self.bg, (0,0))

		#draw elements
		self.pipe_group.draw(self.screen)
		self.bird_group.draw(self.screen)

		#draw ground
		self.screen.blit(self.ground_img, (self.ground_scroll, 768))
		state, _ =  self.get_state()
		return state
	
	def get_score(self):
		return self.score
	
	def action(self, act):
		self.cont_pipe_timer += 1

		#update bird
		self.flappy.update(act)

		#check score
		rwd = 0 #Used for rewarding the agent
		if len(self.pipe_group) > 0:
			if self.bird_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.left\
				and self.bird_group.sprites()[0].rect.right < self.pipe_group.sprites()[0].rect.right\
				and self.pass_pipe == False:
				self.pass_pipe = True
				rwd = 1 #Used for rewarding the agent
			if self.pass_pipe == True:
				if self.bird_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.right:
					self.score += 1
					self.pass_pipe = False

		#look for collision
		if pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False) or self.flappy.rect.top < 0:
			self.game_over = True
		#once the bird has hit the ground it's game over and no longer flying
		if self.flappy.rect.bottom >= 768 or self.flappy.rect.top <= 0:
			self.game_over = True
			self.flying = False

		if self.flying == True and self.game_over == False:

			#generate new pipes
			if self.cont_pipe_timer > self.pipe_frequency:
				self.cont_pipe_timer = 0

				pipe_height = random.randint(-100, 100)
				btm_pipe = self.Pipe(self.screen_width, int(self.screen_height / 2) + pipe_height, -1, self)
				top_pipe = self.Pipe(self.screen_width, int(self.screen_height / 2) + pipe_height, 1, self)
				self.pipe_group.add(btm_pipe)
				self.pipe_group.add(top_pipe)

			self.pipe_group.update()

			self.ground_scroll -= self.scroll_speed
			if abs(self.ground_scroll) > 35:
				self.ground_scroll = 0

		self.screen.blit(self.bg, (0, 0))
		self.pipe_group.draw(self.screen)
		self.bird_group.draw(self.screen)
		self.screen.blit(self.ground_img, (self.ground_scroll, 768))
		self.draw_text(str(self.score), self.font, self.white, int(self.screen_width / 2), 20)

		state, rwd_ = self.get_state()
		
		if(self.show_window):
			pygame.display.update()

		return state, rwd+rwd_ if self.game_over==False else -1, self.game_over

	def close(self):
		pygame.quit()


if __name__ == "__main__":
	hola = FlappyBird()
	hola.reset()
	while(True):
		time.sleep(0.01)
		if hola.action(random.randint(0,1)):
			hola.reset()