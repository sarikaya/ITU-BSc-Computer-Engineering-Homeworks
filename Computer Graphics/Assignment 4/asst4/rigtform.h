#ifndef RIGTFORM_H
#define RIGTFORM_H

#include <iostream>
#include <cassert>

#include "matrix4.h"
#include "quat.h"

class RigTForm {
  Cvec3 t_; // translation component
  Quat r_;  // rotation component represented as a quaternion

public:
	RigTForm() : t_(0) {
		assert(norm2(Quat(1,0,0,0) - r_) < CS175_EPS2);
	}

	RigTForm(const Cvec3& t, const Quat& r) : t_(t), r_(r){
		//TODO (done)
	}

	explicit RigTForm(const Cvec3& t) : t_(t) {
		// TODO (done)
		r_ = Quat();	// identity quaternion
	}

	explicit RigTForm(const Quat& r) : r_(r) {
		// TODO (done)
		t_ = Cvec3();	// identity vector3
	}

	Cvec3 getTranslation() const {
		return t_;
	}

	Quat getRotation() const {
		return r_;
	}

	RigTForm& setTranslation(const Cvec3& t) {
		t_ = t;
		return *this;
	}

	RigTForm& setRotation(const Quat& r) {
		r_ = r;
		return *this;
	}

	Cvec4 operator * (const Cvec4& a) const {
		// TODO (done)
    return r_*a + Cvec4(t_, 0);
	}

	RigTForm operator * (const RigTForm& a) const {
		// TODO (done)

    // convert Cvec3's of RigTForms into Cvec4s
    Cvec4 t4(t_[0], t_[1], t_[2], 1);
    Cvec4 at4(a.t_[0], a.t_[1], a.t_[2], 1);

    // find translational result
    Cvec4 tResult = t4 + r_* at4;
    Cvec3 tResult3(tResult[0], tResult[1], tResult[2]);
    
    // return T = t1 + r1*t2 || R = r1*r2
    return RigTForm(tResult3, r_ * a.r_);
	}
};

inline RigTForm inv(const RigTForm& tform) {
	// TODO (done)

  // make transform component Cvec4
  Cvec4 translation(tform.getTranslation(), 1);
  Cvec4 Tresult = inv(tform.getRotation()) * -translation;

  // convert result back to Cvec3
  Cvec3 Tresult3(Tresult[0], Tresult[1], Tresult[2]);

  // return T = -inv(r)*t || R = inv(r)
  return RigTForm(Tresult3, inv(tform.getRotation()));
  
}

inline RigTForm transFact(const RigTForm& tform) {
  return RigTForm(tform.getTranslation());
}

inline RigTForm linFact(const RigTForm& tform) {
  return RigTForm(tform.getRotation());
}

inline Matrix4 rigTFormToMatrix(const RigTForm& tform) {
	// TODO (done)

  // fix rotational part of matrix4
  Matrix4 m = quatToMatrix(tform.getRotation());
  
  // fix translational part of matrix4
  m(0,3) = tform.getTranslation()[0];
  m(1,3) = tform.getTranslation()[1];
  m(2,3) = tform.getTranslation()[2];

	return m;
}

#endif
