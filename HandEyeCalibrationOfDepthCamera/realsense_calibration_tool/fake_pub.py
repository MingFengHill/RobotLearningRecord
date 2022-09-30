import rospy
from sensor_msgs.msg import JointState


def main():
    JOINT_STATE_TOPIC = "/joint_states"

    pub = rospy.Publisher(JOINT_STATE_TOPIC, JointState, queue_size=10)
    rospy.init_node('fake_pub', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        joint_state = JointState()
        joint_state.name = ["fake1", "fake2", "fake3", "fake4", "fake5", "fake6"]
        joint_state.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        pub.publish(joint_state)
        rate.sleep()


if __name__ == '__main__':
    main()
