import sys	
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import numpy as np 



class DNMC:
	def __init__(self,batch_size,extern_input_size,extern_output_size,memory_address_size,memory_input_size,NM_size,num_read_heads,hidden_size,memory_memory_address_size,memory_memory_input_size,memory_NM_size,):
		self.batch_size=batch_size
		self.extern_input_size=extern_input_size
		self.extern_output_size=extern_output_size
		self.memory_address_size=memory_address_size
		self.memory_input_size=memory_input_size
		self.NM_size=NM_size
		self.num_read_heads=num_read_heads
		self.hidden_size=hidden_size
		self.memory_memory_address_size=memory_memory_address_size
		self.memory_memory_input_size=memory_memory_input_size
		self.memory_NM_size=memory_NM_size

		self.memory_controller_input_size=self.memory_address_size+self.memory_memory_input_size+(self.memory_memory_input_size*self.memory_NM_size+2*self.memory_NM_size*self.memory_NM_size+self.memory_NM_size*self.memory_memory_address_size)
		self.memory_controller_output_size=self.memory_input_size+self.memory_memory_address_size+2*(self.memory_memory_input_size*self.memory_NM_size+2*self.memory_NM_size*self.memory_NM_size+self.memory_NM_size*self.memory_memory_address_size)
		self.controller_input_size=self.extern_input_size+self.memory_input_size*self.num_read_heads+self.memory_controller_input_size*self.NM_size+2*self.NM_size*self.NM_size+self.NM_size*self.memory_controller_output_size
                self.controller_output_size=self.extern_output_size+self.memory_address_size*self.num_read_heads+2*(self.memory_controller_input_size*self.NM_size+2*self.NM_size*self.NM_size+self.NM_size*self.memory_controller_output_size)

		#dynamic Constants
		self.extern_input=tf.placeholder(tf.float64,[self.batch_size,None,self.extern_input_size])
		self.extern_output=tf.placeholder(tf.float64,[self.batch_size,None,self.extern_output_size])
		self.time_steps=tf.placeholder(tf.int32)

		self.memory_input=tf.zeros([self.batch_size,self.memory_input_size*self.num_read_heads],dtype=tf.float64)
		
		self.memory_layer1=tf.zeros([self.batch_size,self.memory_controller_input_size,self.NM_size],dtype=tf.float64)
		self.memory_layer2=tf.zeros([self.batch_size,self.NM_size,self.NM_size],dtype=tf.float64)
		self.memory_layer3=tf.zeros([self.batch_size,self.NM_size,self.NM_size],dtype=tf.float64)
		self.memory_layer4=tf.zeros([self.batch_size,self.NM_size,self.memory_controller_output_size],dtype=tf.float64)
		
		#self.read_weight=tf.zeros([self.batch_size,self.memory_address_size],dtype=tf.float64)
		
		#self.write_weight1=tf.zeros([self.batch_size,self.memory_address_size*self.NM_size],dtype-tf.float64)
		#self.write_weight2=tf.zeros([self.batch_size,self.NM_size*self.NM_size],dtype=tf.float64)
		#self.write_weight3=tf.zeros([self.batch_size,self.NM_size*self.NM_size],dtype=tf.float64)
		#self.write_weight4=tf.zeros([self.batch_size,self.NM_size*self.memory_input_size],dtype=tf.float64)
		
		self.extern_output_time=tf.zeros([self.batch_size,0,self.extern_output_size],dtype=tf.float64)
		self.i=tf.constant(0)



		#static Constants
		self.sequence_length=tf.ones([self.batch_size],tf.int64)

		#Controller Variables

		"""self.controller_output_lstm=rnn_cell.BasicLSTMCell(self.controller_output_size,state_is_tuple=False)
                cell=[]
                for i in xrange(self.num_controller_layers):
                        controller_lstm=rnn_cell.BasicLSTMCell(self.hidden_size,state_is_tuple=False)
                        cell.append(controller_lstm)
                cell.append(self.controller_output_lstm)

                self.controller=tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=False)

                self.controller_state=self.controller.zero_state(self.batch_size,tf.float64)"""


		self.controller_layer1=tf.Variable(tf.random_normal([self.controller_input_size,self.hidden_size],dtype=tf.float64))
		self.controller_layer2=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
		self.controller_layer3=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
		self.controller_layer4=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
		self.controller_layer5=tf.Variable(tf.random_normal([self.hidden_size,self.hidden_size],dtype=tf.float64))
		self.controller_layer6=tf.Variable(tf.random_normal([self.hidden_size,self.controller_output_size],dtype=tf.float64))


		self.controller_bias1=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
		self.controller_bias2=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
		self.controller_bias3=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
		self.controller_bias4=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
		self.controller_bias5=tf.Variable(tf.random_normal([self.hidden_size],dtype=tf.float64))
		self.controller_bias6=tf.Variable(tf.random_normal([self.controller_output_size],dtype=tf.float64))



		#Computational Graph



		while_loop_output=tf.while_loop(self.DNMC_while_condition,self.DNMC_while_loop,\
		[self.i,self.extern_input,self.extern_output_time,self.memory_input,self.memory_layer1,self.memory_layer2,self.memory_layer3,self.memory_layer4],\
		[self.i.get_shape(),self.extern_input.get_shape(),tf.TensorShape([self.batch_size,None,self.extern_output_size]),tf.TensorShape([self.batch_size,None]),self.memory_layer1.get_shape(),\
		self.memory_layer2.get_shape(),self.memory_layer3.get_shape(),self.memory_layer4.get_shape()])


		_,_,self.extern_output_time,_,_,_,_,_=while_loop_output


		self.cost=tf.reduce_sum(tf.square(self.extern_output_time-self.extern_output))

		self.optimizer=tf.train.AdamOptimizer(0.0001).minimize(self.cost)


	def Controller(self,extern_input,memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
		memory_layer1=tf.reshape(memory_layer1,[self.batch_size,self.memory_controller_input_size*self.NM_size])
		memory_layer2=tf.reshape(memory_layer2,[self.batch_size,self.NM_size*self.NM_size])
		memory_layer3=tf.reshape(memory_layer3,[self.batch_size,self.NM_size*self.NM_size])
		memory_layer4=tf.reshape(memory_layer4,[self.batch_size,self.NM_size*self.memory_controller_output_size])
		controller_input=tf.concat([extern_input,memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4],axis=1)
		
		"""controller_input=tf.reshape(controller_input,[self.batch_size,1,self.controller_input_size])
		controller_output,controller_state=tf.nn.dynamic_rnn(self.controller,controller_input,sequence_length=self.sequence_length,initial_state=controller_state)
		controller_output=tf.reshape(controller_output,[self.batch_size,self.controller_output_size])"""

		controller_output=tf.sigmoid(tf.add(tf.matmul(controller_input,self.controller_layer1),self.controller_bias1))
		controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer2),self.controller_bias2))
		controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer3),self.controller_bias3))
		#controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer4),self.controller_bias4))
		#controller_output=tf.sigmoid(tf.add(tf.matmul(controller_output,self.controller_layer5),self.controller_bias5))
		controller_output=tf.add(tf.matmul(controller_output,self.controller_layer6),self.controller_bias6)

		extern_output,read_weight,write_weight1,erase_weight1,write_weight2,erase_weight2,write_weight3,erase_weight3,write_weight4,erase_weight4=tf.split(controller_output,[self.extern_output_size,self.memory_address_size*self.num_read_heads,\
		self.memory_controller_input_size*self.NM_size,self.memory_controller_input_size*self.NM_size,self.NM_size*self.NM_size,self.NM_size*self.NM_size,self.NM_size*self.NM_size,self.NM_size*self.NM_size,\
		self.NM_size*self.memory_controller_output_size,self.NM_size*self.memory_controller_output_size],axis=1)
		extern_output=extern_output
		read_weight=tf.sigmoid(read_weight)
		write_weight1=tf.sigmoid(write_weight1)
		write_weight2=tf.sigmoid(write_weight2)
		write_weight3=tf.sigmoid(write_weight3)
		write_weight4=tf.sigmoid(write_weight4)

		erase_weight1=tf.sigmoid(erase_weight1)
		erase_weight2=tf.sigmoid(erase_weight2)
		erase_weight3=tf.sigmoid(erase_weight3)
		erase_weight4=tf.sigmoid(erase_weight4)
		



		return extern_output,read_weight,write_weight1,write_weight2,write_weight3,write_weight4,erase_weight1,erase_weight2,erase_weight3,erase_weight4
	
	
	def Memory_Memory_Access(self,memory_read_weight,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4):
		memory_read_weight=tf.reshape(memory_read_weight,[self.batch_size,1,self.memory_memory_address_size])
		memory_memory_controller_output=tf.sigmoid(tf.matmul(memory_read_weight,memory_memory_layer1))
		memory_memory_controller_output=tf.sigmoid(tf.matmul(memory_memory_controller_output,memory_memory_layer2))
		memory_memory_controller_output=tf.sigmoid(tf.matmul(memory_memory_controller_output,memory_memory_layer3))
		memory_memory_controller_output=tf.sigmoid(tf.matmul(memory_memory_controller_output,memory_memory_layer4))
		memory_memory_controller_output=tf.reshape(memory_memory_controller_output,[self.batch_size,self.memory_memory_input_size])
		return memory_memory_controller_output		


	def Memory_Access(self,read_weight,memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4):
		memory_memory_layer1_=tf.reshape(memory_memory_layer1,[self.batch_size,self.memory_memory_address_size*self.memory_NM_size])
		memory_memory_layer2_=tf.reshape(memory_memory_layer2,[self.batch_size,self.memory_NM_size*self.memory_NM_size])
		memory_memory_layer3_=tf.reshape(memory_memory_layer3,[self.batch_size,self.memory_NM_size*self.memory_NM_size])
		memory_memory_layer4_=tf.reshape(memory_memory_layer4,[self.batch_size,self.memory_NM_size*self.memory_memory_input_size])

		memory_controller_input=tf.concat([read_weight,memory_memory_input,memory_memory_layer1_,memory_memory_layer2_,memory_memory_layer3_,memory_memory_layer4_],axis=1)
		memory_controller_input=tf.reshape(memory_controller_input,[self.batch_size,1,self.memory_controller_input_size])
		memory_controller_output=tf.sigmoid(tf.matmul(memory_controller_input,memory_layer1))
		memory_controller_output=tf.sigmoid(tf.matmul(memory_controller_output,memory_layer2))
		memory_controller_output=tf.sigmoid(tf.matmul(memory_controller_output,memory_layer3))
		memory_controller_output=tf.matmul(memory_controller_output,memory_layer4)
		
		memory_controller_output=tf.reshape(memory_controller_output,[self.batch_size,self.memory_controller_output_size])

		memory_input,memory_read_weight,memory_write_weight1,memory_write_weight2,memory_write_weight3,memory_write_weight4,memory_erase_weight1,memory_erase_weight2,memory_erase_weight3,memory_erase_weight4=\
		tf.split(memory_controller_output,\
		[self.memory_input_size,self.memory_memory_address_size,self.memory_memory_address_size*self.memory_NM_size,self.memory_NM_size*self.memory_NM_size,self.memory_NM_size*self.memory_NM_size,self.memory_NM_size*self.memory_memory_input_size,\
		self.memory_memory_address_size*self.memory_NM_size,self.memory_NM_size*self.memory_NM_size,self.memory_NM_size*self.memory_NM_size,self.memory_NM_size*self.memory_memory_input_size],axis=1)
		

		memory_input=tf.sigmoid(memory_input)
		memory_read_weight=tf.sigmoid(memory_read_weight)
		memory_write_weight1=tf.sigmoid(memory_write_weight1)
		memory_write_weight2=tf.sigmoid(memory_write_weight2)
		memory_write_weight3=tf.sigmoid(memory_write_weight3)
		memory_write_weight4=tf.sigmoid(memory_write_weight4)
	
		memory_erase_weight1=tf.sigmoid(memory_erase_weight1)
		memory_erase_weight2=tf.sigmoid(memory_erase_weight2)
		memory_erase_weight3=tf.sigmoid(memory_erase_weight3)
		memory_erase_Weight4=tf.sigmoid(memory_erase_weight4)

		memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4=self.Memory_Memory_Change(memory_write_weight1,memory_write_weight2,memory_write_weight3,memory_write_weight4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4)
		memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4=self.Memory_Memory_Change2(memory_erase_weight1,memory_erase_weight2,memory_erase_weight3,memory_erase_weight4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4)

		memory_memory_input=self.Memory_Memory_Access(memory_read_weight,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4)
		


		
	
		return memory_input,memory_memory_input,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4

	def Memory_Change(self,write_weight1,write_weight2,write_weight3,write_weight4,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
		write_weight1=tf.reshape(write_weight1,[self.batch_size,self.memory_controller_input_size,self.NM_size])
		write_weight2=tf.reshape(write_weight2,[self.batch_size,self.NM_size,self.NM_size])
		write_weight3=tf.reshape(write_weight3,[self.batch_size,self.NM_size,self.NM_size])
		write_weight4=tf.reshape(write_weight4,[self.batch_size,self.NM_size,self.memory_controller_output_size])

		memory_layer1=tf.add(memory_layer1,write_weight1)
		memory_layer2=tf.add(memory_layer2,write_weight2)
		memory_layer3=tf.add(memory_layer3,write_weight3)
		memory_layer4=tf.add(memory_layer4,write_weight4)

		return memory_layer1,memory_layer2,memory_layer3,memory_layer4

	def Memory_Memory_Change(self,memory_write_weight1,memory_write_weight2,memory_write_weight3,memory_write_weight4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4):
                memory_write_weight1=tf.reshape(memory_write_weight1,[self.batch_size,self.memory_memory_address_size,self.memory_NM_size])
                memory_write_weight2=tf.reshape(memory_write_weight2,[self.batch_size,self.memory_NM_size,self.memory_NM_size])
                memory_write_weight3=tf.reshape(memory_write_weight3,[self.batch_size,self.memory_NM_size,self.memory_NM_size])
                memory_write_weight4=tf.reshape(memory_write_weight4,[self.batch_size,self.memory_NM_size,self.memory_memory_input_size])

                memory_memory_layer1=tf.add(memory_memory_layer1,memory_write_weight1)
                memory_memory_layer2=tf.add(memory_memory_layer2,memory_write_weight2)
                memory_memory_layer3=tf.add(memory_memory_layer3,memory_write_weight3)
                memory_memory_layer4=tf.add(memory_memory_layer4,memory_write_weight4)

                return memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4



	def Memory_Change2(self,erase_weight1,erase_weight2,erase_weight3,erase_weight4,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
                erase_weight1=tf.reshape(erase_weight1,[self.batch_size,self.memory_controller_input_size,self.NM_size])
                erase_weight2=tf.reshape(erase_weight2,[self.batch_size,self.NM_size,self.NM_size])
                erase_weight3=tf.reshape(erase_weight3,[self.batch_size,self.NM_size,self.NM_size])
                erase_weight4=tf.reshape(erase_weight4,[self.batch_size,self.NM_size,self.memory_controller_output_size])

                memory_layer1=tf.sigmoid(tf.subtract(memory_layer1,erase_weight1))
                memory_layer2=tf.sigmoid(tf.subtract(memory_layer2,erase_weight2))
                memory_layer3=tf.sigmoid(tf.subtract(memory_layer3,erase_weight3))
                memory_layer4=tf.sigmoid(tf.subtract(memory_layer4,erase_weight4))

                return memory_layer1,memory_layer2,memory_layer3,memory_layer4

	def Memory_Memory_Change2(self,erase_weight1,erase_weight2,erase_weight3,erase_weight4,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
                erase_weight1=tf.reshape(erase_weight1,[self.batch_size,self.memory_memory_address_size,self.memory_NM_size])
                erase_weight2=tf.reshape(erase_weight2,[self.batch_size,self.memory_NM_size,self.memory_NM_size])
                erase_weight3=tf.reshape(erase_weight3,[self.batch_size,self.memory_NM_size,self.memory_NM_size])
                erase_weight4=tf.reshape(erase_weight4,[self.batch_size,self.memory_NM_size,self.memory_memory_input_size])

                memory_layer1=tf.sigmoid(tf.subtract(memory_layer1,erase_weight1))
                memory_layer2=tf.sigmoid(tf.subtract(memory_layer2,erase_weight2))
                memory_layer3=tf.sigmoid(tf.subtract(memory_layer3,erase_weight3))
                memory_layer4=tf.sigmoid(tf.subtract(memory_layer4,erase_weight4))

                return memory_layer1,memory_layer2,memory_layer3,memory_layer4


	def DNMC_while_loop(self,i,extern_input,extern_output_time,memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
		
		extern_output,read_weight,write_weight1,write_weight2,write_weight3,write_weight4,erase_weight1,erase_weight2,erase_weight3,erase_weight4=self.Controller(extern_input[:,i,:],memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4)
		
		extern_output_time=tf.concat([extern_output_time,tf.reshape(extern_output,[self.batch_size,1,self.extern_output_size])],axis=1)


		memory_layer1,memory_layer2,memory_layer3,memory_layer4=self.Memory_Change(write_weight1,write_weight2,write_weight3,write_weight4,memory_layer1,memory_layer2,memory_layer3,memory_layer4)

		memory_layer1,memory_layer2,memory_layer3,memory_layer4=self.Memory_Change2(erase_weight1,erase_weight2,erase_weight3,erase_weight4,memory_layer1,memory_layer2,memory_layer3,memory_layer4)

		j=tf.constant(0)
		memory_input=tf.zeros([self.batch_size,0],dtype=tf.float64)
		memory_memory_input=tf.zeros([self.batch_size,self.memory_memory_input_size],dtype=tf.float64)
		memory_memory_layer1=tf.zeros([self.batch_size,self.memory_memory_address_size,self.memory_NM_size],dtype=tf.float64)
		memory_memory_layer2=tf.zeros([self.batch_size,self.memory_NM_size,self.memory_NM_size],dtype=tf.float64)
		memory_memory_layer3=tf.zeros([self.batch_size,self.memory_NM_size,self.memory_NM_size],dtype=tf.float64)
		memory_memory_layer4=tf.zeros([self.batch_size,self.memory_NM_size,self.memory_memory_input_size],dtype=tf.float64)

		_,_,memory_input,_,_,_,_,_,_,_,_,_=tf.while_loop(self.Memory_Access_while_condition,self.Memory_Access_while_loop,\
		[j,read_weight,memory_input,memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4],\
		[j.get_shape(),read_weight.get_shape(),tf.TensorShape([self.batch_size,None]),memory_memory_input.get_shape(),memory_layer1.get_shape(),memory_layer2.get_shape(),memory_layer3.get_shape(),memory_layer4.get_shape(),\
		memory_memory_layer1.get_shape(),memory_memory_layer2.get_shape(),memory_memory_layer3.get_shape(),memory_memory_layer4.get_shape()])

		i=tf.add(i,1)

		return i,extern_input,extern_output_time,memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4
	
		
	def DNMC_while_condition(self,i,extern_output,extern_output_time,memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4):
		return i<self.time_steps

	def Memory_Access_while_loop(self,j,read_weight,memory_input,memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4):
		memory_input_,memory_memory_input,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4=\
		self.Memory_Access(read_weight[:,self.memory_address_size*j:self.memory_address_size*(j+1)],memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4)
		memory_input=tf.concat([memory_input,memory_input_],axis=1)
		j=tf.add(j,1)

		return j,read_weight,memory_input,memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4

	def Memory_Access_while_condition(self,j,read_weight,memory_input,memory_memory_input,memory_layer1,memory_layer2,memory_layer3,memory_layer4,memory_memory_layer1,memory_memory_layer2,memory_memory_layer3,memory_memory_layer4):
		return tf.less(j,tf.constant(self.num_read_heads))

	def train(self,train_steps,path_to_tb_dir):
                saver=tf.train.Saver()

                sess=tf.Session()
        #       writer=tf.summary.FileWriter(path_to_tb_dir,sess.graph)
                sess.run(tf.global_variables_initializer())

                for i in xrange(train_steps):
                        time_steps=20
                        #time_steps=np.random.randint(low=10,high=25)
                        batch_x=np.empty((0,time_steps,self.extern_input_size),dtype="float64")
                        batch_y=np.empty((0,time_steps,self.extern_output_size),dtype="float64")

                        for j in xrange(self.batch_size):
				
                                random_vector=np.random.rand(time_steps-5,self.extern_output_size).astype('float64')

                                batch_x=np.append(batch_x,np.append(random_vector,np.zeros((5,self.extern_output_size)),axis=0).reshape(1,time_steps,self.extern_input_size),axis=0)
                                batch_y=np.append(batch_y,np.append(np.zeros((5,self.extern_output_size)),random_vector,axis=0).reshape(1,time_steps,self.extern_output_size),axis=0)
                                
                                        
			#summary=tf.Summary(summary)
                        _,c,pre=sess.run([self.optimizer,self.cost,self.extern_output_time],feed_dict={self.extern_input:batch_x,self.extern_output:batch_y,self.time_steps:time_steps})
                        #writer.add_summary(summary)
			print(batch_x[0])
			print pre[0]
			print pre[0]-batch_y[0]
			print i,'th',c 
        #       writer.close()

                saver.save(sess,"/root/python_programs/ntm")



DNMC=DNMC(batch_size=200,extern_input_size=1,extern_output_size=1,memory_address_size=6,memory_input_size=11,NM_size=8,num_read_heads=6,hidden_size=101,memory_memory_address_size=1,memory_memory_input_size=1,memory_NM_size=2)
	
DNMC.train(10000,"/path/to/your/tb/dir")

