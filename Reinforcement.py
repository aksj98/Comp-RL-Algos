import streamlit as st
import gym
import subprocess


st.title("Comparison of different reinforcement learning algorithms")

option=st.selectbox("Which type of algorithms would you like to compare",["On-Policy(LunarLander)","Off-Policy(Pendulum)"])

if (option=="On-Policy(LunarLander)"):
	st.title("Algorithm One")
	alge1=""
	algo1=st.selectbox("Type of policy",["VPG","TRPO","PPO"])
	if (algo1=="VPG"):
		alge1="vpg-pyt-bench_lunarlander-v2_"
	elif (algo1=="TRPO"):
		alge1="trpo-tf1-bench_lunarlander-v2_"
	else:
		alge1="ppo-pyt-bench_lunarlander-v2_"
	algo11=st.selectbox("Hidden layers",["32","64-64"
])
	if (algo11=="32"):
		alge1=alge1+"hid32_"
	else:
		alge1=alge1+"hid64-64_"
	algo12=st.selectbox("Activation",["tanh","relu"])
	if (algo12=="tanh"):
		alge1=alge1+"tanh"
	else:

		alge1=alge1+"relu"
	st.title("Algorithm Two")
	alge2=""
	algo2=st.selectbox("Type of policy",["Vanilla Policy Gradient","Trust Region Policy Optimization","Proximal Policy Optimization"])
	if (algo2=="Vanilla Policy Gradient"):
		alge2="vpg-pyt-bench_lunarlander-v2_"
	elif (algo2=="Trust Region Policy Optimization"):
		alge2="trpo-tf1-bench_lunarlander-v2_"
	else:
		alge2="ppo-pyt-bench_lunarlander-v2_"
	algo21=st.selectbox("Hidden layers",["32","64"
])
	if (algo21=="32"):
		alge2=alge2+"hid32_"
	else:
		alge2=alge2+"hid64-64_"
	algo22=st.selectbox("Activation",["tanH","relu"])
	if (algo22=="tanH"):
		alge2=alge2+"tanh"
	else:

		alge2=alge2+"relu"
	if st.button("Compare"):
		subprocess.call(["python","-m","spinup.run","plot","./data/onpolicy/"+alge1,"./data/onpolicy/"+alge2])
		st.write("Click on Compare to generate a graph comparing the two policies")
else:
	st.title("Algorithm One")
	alge1=""
	alg1=st.selectbox("Type of policy",["DDPG","TD3","SAC"])
	if (alg1=="DDPG"):
		alge1="ddpg-pyt-bench_pendulum-v0_"
	elif (alg1=="TD3"):
		alge1="td3-pyt-bench_pendulum-v0_"
	else:
		alge1="sac-pyt-bench_pendulum-v0_"
	alg11=st.selectbox("Hidden layers",["32","64-64"
])
	if (alg11=="32"):
		alge1=alge1+"hid32_"
	else:
		alge1=alge1+"hid64-64_"
	alg12=st.selectbox("Activation",["tanh","relu"])
	if (alg12=="tanh"):
		alge1=alge1+"tanh"
	else:

		alge1=alge1+"relu"
	st.title("Algorithm Two")
	alge2=""
	
	alg2=st.selectbox("Type of policy",["DDPG","TD3G","SAC"])
	if (alg2=="DDPG"):
		alge2="ddpg-pyt-bench_pendulum-v0_"
	elif (alg2=="TD3G"):
		alge2="td3-pyt-bench_pendulum-v0_"
	else:
		alge2="sac-pyt-bench_pendulum-v0_"
	alg21=st.selectbox("Hidden layers",["32","64"
])
	if (alg21=="32"):
		alge2=alge2+"hid32_"
	else:
		alge2=alge2+"hid64-64_"
	alg22=st.selectbox("Activation",["tanH","relu"])
	if (alg22=="tanH"):
		alge2=alge2+"tanh"
	else:

		alge2=alge2+"relu"
	if st.button("Compare"):
		subprocess.call(["python","-m","spinup.run","plot","./data/off-policy/"+alge1,"./data/off-policy/"+alge2])
		st.write("Click on Compare to compare the two policies")
