////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <vector>
#include <list>
#include <string>
#include <fstream>
#include <memory>
#include <stdexcept>
#if __GNUG__
#   include <tr1/memory>
#endif

#include <GL/glew.h>
#ifdef __MAC__
#   include <GLUT/glut.h>
#else
#   include <GL/glut.h>
#endif

#include "cvec.h"
#include "matrix4.h"
#include "geometrymaker.h"
#include "ppm.h"
#include "glsupport.h"
#include "rigtform.h"
#include "quat.h"
#include "arcball.h"
#include "asstcommon.h"
#include "scenegraph.h"
#include "drawer.h"
#include "picker.h"
#include "sgutils.h"

using namespace std;      // for string, vector, iostream, and other standard C++ stuff
using namespace tr1; // for shared_ptr

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.3.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.3. Make sure that your machine supports the version of GLSL you
// are using. In particular, on Mac OS X currently there is no way of using
// OpenGL 3.x with GLSL 1.3 when GLUT is used.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;

static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

// states: 0=eye, 1=redcube, 2=greencube
static unsigned viewState = 0;
static unsigned manipState = 1;
static bool worldSkyFrame = true;

// arcball
static float g_arcballScreenRadius = 0.25*min(g_windowWidth, g_windowHeight);
static float g_arcballScale;

// shaders
static const int PICKING_SHADER = 2; // index of the picking shader is g_shaerFiles
static const int g_numShaders = 3; // 3 shaders instead of 2
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"}
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/pick-gl2.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states

// --------- Geometry

////////////////////// Animation variables
vector<shared_ptr<SgRbtNode> > rbtNodes;
list<vector<RigTForm> > keyframes;
list<vector<RigTForm> >::iterator currentFrame = keyframes.end(); //undefined initially

static int g_msBetweenKeyFrames = 2000;
static int g_animateFramesPerSecond = 60;

bool isPlaying = false;

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) &(((StructType *)0)->field)

// A vertex with floating point position and normal
struct VertexPN {
  Cvec3f p, n;

  VertexPN() {}
  VertexPN(float x, float y, float z,
           float nx, float ny, float nz)
    : p(x,y,z), n(nx, ny, nz)
  {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPN(const GenericVertex& v) {
    *this = v;
  }

  VertexPN& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  int vboLen, iboLen;

  Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);
  }
};

typedef SgGeometryShapeNode<Geometry> MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;
static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot1Node, g_robot2Node;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode; // used later when you do picking

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space

static RigTForm g_skyRbt = RigTForm( Cvec3(0.0, 0.25, 4.0) );
//static Matrix4 g_skyRbt = Matrix4::makeTranslation(Cvec3(0.0, 0.25, 4.0));

static RigTForm g_objRbt[2] = { RigTForm( Cvec3(-.85,0,0) ), RigTForm( Cvec3(.85, 0, 0) ) };
//static Matrix4 g_objectRbt[2] = {Matrix4::makeTranslation(Cvec3(-.85,0,0)), Matrix4::makeTranslation(Cvec3(.85,0,0))};  // currently only 1 obj is defined

static Cvec3f g_objectColors[3] = {Cvec3f(1, 0, 0), Cvec3f(.051,.75,.137), Cvec3f(0, 0, 1)};

static RigTForm g_sphereRbt = RigTForm();

///////////////// END OF G L O B A L S //////////////////////////////////////////////////




static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}

static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);

  // Temporary storage for cube geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere(){
  int ibLen, vbLen;
  getSphereVbIbLen(20, 20, vbLen, ibLen);

  // Temporary storage for sphere geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeSphere(1, 20, 20, &vtx[0], &idx[0]);
  g_sphere.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

// takes a projection matrix and send to the the shaders
static void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}

static void drawStuff(const ShaderState& curSS, bool picking) {

  // build & send proj. matrix to vshader
  const Matrix4 projmat = makeProjectionMatrix();
  sendProjectionMatrix(curSS, projmat);

  // assign eye rigid body matrix;
  //Matrix4 eyeRbt;
  RigTForm eyeRbt;
  switch(viewState){
  case 0:
	  eyeRbt = g_skyNode->getRbt();
	  break;
  case 1:                                     
    eyeRbt = g_robot1Node->getRbt();
    break;
  case 2:       
	  eyeRbt = g_robot2Node->getRbt();
	  break;
  }
  
  const RigTForm invEyeRbt = inv(eyeRbt);

  const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1)); // g_light1 position in eye coordinates
  const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1)); // g_light2 position in eye coordinates
  safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
  safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);

  if (!picking) {
    Drawer drawer(invEyeRbt, curSS);
    g_world->accept(drawer);

  // draw arcball as part of asst3
  // fix scale and you're good to go
  //g_arcballScale = getScreenToEyeScale(0, g_frustFovY, g_windowHeight);
  g_arcballScale = 1; 
  /*
  switch(manipState){
  case 0:                                  
    g_sphereRbt.setTranslation(Cvec3(0,0,0));
    break;
  case 1:
    g_sphereRbt.setTranslation(g_objRbt[0].getTranslation());
    g_sphereRbt.setRotation(g_objRbt[0].getRotation());
    break;
  case 2:                                                    
    g_sphereRbt.setTranslation(g_objRbt[1].getTranslation());
    g_sphereRbt.setRotation(g_objRbt[1].getRotation());
    break;
  }

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);  // draw wireframe

  // compute MVM, taking into account the dynamic radius
  float scale =  g_arcballScale * g_arcballScreenRadius;
  Matrix4 MVM = rigTFormToMatrix(inv(eyeRbt) * g_sphereRbt) * g_arcballScale;
  Matrix4 NMVM = normalMatrix(MVM);

  // send MVM and NMVM and colors
  sendModelViewNormalMatrix(curSS, MVM, NMVM);
  safe_glUniform3f(curSS.h_uColor, g_objectColors[2][0], g_objectColors[2][1], g_objectColors[2][2]);
  g_sphere->draw(curSS);

  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  */
  }
  else {
    Picker picker(invEyeRbt, curSS);
    g_world->accept(picker);
    glFlush();
    g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
    if (g_currentPickedRbtNode == g_groundNode)
      g_currentPickedRbtNode = shared_ptr<SgRbtNode>();   // set to NULL
  }

  // draw arcball
  // ============
  


}

static void display() {
  glUseProgram(g_shaderStates[g_activeShader]->program);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);                   // clear framebuffer color&depth

  drawStuff(*g_shaderStates[g_activeShader], false);

  glutSwapBuffers();                                    // show the back buffer (where we rendered stuff)

  checkGlErrors();
}

static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  g_arcballScreenRadius = 0.25*min(g_windowWidth, g_windowHeight);
  updateFrustFovY();
  glutPostRedisplay();
}

static void doMtoOwrtA(Matrix4 m, Matrix4& o, Matrix4 a){
	Matrix4 A = transFact(o)*linFact(a);
	o = A * m * inv(A) * o;
}

static void doMtoOwrtA(RigTForm m, RigTForm& o, RigTForm a){
	RigTForm A = transFact(o)*linFact(a);
	o = A * m * inv(A) * o;
}
                                         
static void doMtoOwrtA(RigTForm m, shared_ptr<SgRbtNode>& o, shared_ptr<SgRbtNode>& a){
  RigTForm O = o->getRbt();
  RigTForm AA = a->getRbt();
  RigTForm A = transFact(O)*linFact(AA);
  o->setRbt(A * m * inv(A) * O);
}

static void noInterfaceRotation(RigTForm& m){
		switch(manipState){ // manipulation state: 0=sky, 1=redCube, 2=greenCube
		case 0:		// sky camera is manipulated
			if(viewState == 0){		// sky camera is the eye
        if(worldSkyFrame){  // if world-sky frame
          RigTForm A = transFact(m) * linFact(inv(g_skyNode->getRbt()));

          //doMtoOwrtA(inv(m), g_skyRbt, A);
          // here we cannot doMtoOwrtA since the function takes transfact(#2 param)
          // which corresponds to transfact(g_skyRbt) while we prefer transfact(m)
          // therefore we need to define our frame and do the multiplication on our own
          // in order to achieve circulating motion around the world frame.

          // translation is inverted when fed to the function
          RigTForm O = g_skyNode->getRbt();
          g_skyNode->setRbt(A*inv(m)*inv(A)*O);
        }
        else{    // if sky-sky frame                    
          // translation is inverted before feeding into function
          // therefore it will be inverted back to normal when fed to function
          // thus only the rotations will be inverted
          m = transFact(inv(m)) * linFact(m);  
          doMtoOwrtA(inv(m), g_skyNode, g_skyNode); 
        }
      }
			break;
		case 1:		// red robot is manipulated
      switch(viewState){
      case 0:   // eye = sky-camera
        doMtoOwrtA(m, g_robot1Node, g_skyNode);	// cube-sky frame
        break;
      case 1:   // eye = red-robot itself    
        m = transFact(inv(m)) * linFact(m);  // keep translation un-inverted
        doMtoOwrtA(inv(m), g_robot1Node, g_robot1Node); 
        break;
      case 2:   // eye = blue-robot      
        doMtoOwrtA(m, g_robot1Node, g_robot2Node);	//red robot - blue robot frame
        break;
      }
			break;
		case 2:		// green robot is manipulated
			switch(viewState){
      case 0:   // eye = sky-camera
        doMtoOwrtA(m, g_robot2Node, g_skyNode);	// cube-sky frame
        break;
      case 1:   // eye = red-robot 
        doMtoOwrtA(m, g_robot2Node, g_robot1Node); 
        break;
      case 2:   // eye = blue-robot itself
        m = transFact(inv(m)) * linFact(m);  // keep translation un-inverted
        doMtoOwrtA(inv(m), g_robot2Node, g_robot2Node);	
        break;
      }
			break;
		}
}

static void arcBallRotation(RigTForm& m){
   	switch(manipState){ // manipulation state: 0=sky, 1=redCube, 2=greenCube
		case 0:		// sky camera is manipulated
			if(viewState == 0){		// sky camera is the eye
        if(worldSkyFrame){  // if world-sky frame
          //Matrix4 A = transFact(m)*linFact(inv(g_skyRbt));
          RigTForm A = transFact(m) * linFact(inv(g_skyRbt));

          //doMtoOwrtA(inv(m), g_skyRbt, A);
          // here we cannot doMtoOwrtA since the function takes transfact(#2 param)
          // which corresponds to transfact(g_skyRbt) while we prefer transfact(m)
          // therefore we need to define our frame and do the multiplication on our own
          // in order to achieve circulating motion around the world frame.

          // translation is inverted when fed to the function
          g_skyRbt = A*inv(m)*inv(A)*g_skyRbt;
        }
        else{    // if sky-sky frame                    
          // translation is inverted before feeding into function
          // therefore it will be inverted back to normal when fed to function
          // thus only the rotations will be inverted
          m = transFact(inv(m)) * linFact(m);  
          doMtoOwrtA(inv(m), g_skyRbt, g_skyRbt); 
        }
      }
			break;
		case 1:		// red cube is manipulated
      switch(viewState){
      case 0:   // eye = sky-camera
        doMtoOwrtA(m, g_objRbt[0], g_skyRbt);	// cube-sky frame
        break;
      case 1:   // eye = red-cube itself    
        m = transFact(inv(m)) * linFact(m);  // keep translation un-inverted
        doMtoOwrtA(inv(m), g_objRbt[0], g_objRbt[0]); 
        break;
      case 2:   // eye = green-cube      
        doMtoOwrtA(m, g_objRbt[0], g_objRbt[1]);	//cube i - cube j frame
        break;
      }
			break;
		case 2:		// green cube is manipulated
			switch(viewState){
      case 0:   // eye = sky-camera
        doMtoOwrtA(m, g_objRbt[1], g_skyRbt);	// cube-sky frame
        break;
      case 1:   // eye = red-cube 
        doMtoOwrtA(m, g_objRbt[1], g_objRbt[0]); 
        break;
      case 2:   // eye = green-cube itself
        m = transFact(inv(m)) * linFact(m);  // keep translation un-inverted
        doMtoOwrtA(inv(m), g_objRbt[1], g_objRbt[1]);	//cube i - cube j frame
        break;
      }
			break;
		}
}

static void motion(const int x, const int y) {
	const double dx = x - g_mouseClickX;
	const double dy = g_windowHeight - y - 1 - g_mouseClickY;

	//Matrix4 m;
  RigTForm m;
	if (g_mouseLClickButton && !g_mouseRClickButton) { // left button down?
    m = m.setRotation( Quat::makeXRotation(-dy) * Quat::makeYRotation(dx) );
    
	}
	else if (g_mouseRClickButton && !g_mouseLClickButton) { // right button down?
    m = m.setTranslation(Cvec3(dx, dy, 0) * 0.01);
	}
	else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton)) {  // middle or (left and right) button down?
    m = m.setTranslation(Cvec3(0, 0, -dy) * 0.01);
	}

	if (g_mouseClickDown) {

    noInterfaceRotation(m);
    //arcBallRotation(m);

		glutPostRedisplay(); // we always redraw if we changed the scene
	}

	g_mouseClickX = x;
	g_mouseClickY = g_windowHeight - y - 1;
}


static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;
}

void writeOut(const list<vector<RigTForm> >& keyframes){
  /*
    File format: Name.txt
    6         // keyframe count
    22        // RigTForm count per keyframe
    x y z     // translation
    w x y z   // rotation (quaternion)
    x y z
    w x y z
    .
    .
    .
  */

  ofstream out("out.txt");
  if(out.is_open()){
    out << keyframes.size() << endl;
    if(keyframes.size() > 0)
      out << keyframes.front().size() << endl;

    for(list<vector<RigTForm> >::const_iterator it = keyframes.begin(); it != keyframes.end() ; ++it){
      for(vector<RigTForm>::const_iterator ite = it->begin(); ite != it->end() ; ite++){
        out << ite->getTranslation()[0] << " " << ite->getTranslation()[1] << " " << ite->getTranslation()[2] << endl;
        out << ite->getRotation()[0] << " " << ite->getRotation()[1] << " " << ite->getRotation()[2] << " " << ite->getRotation()[3] << endl;
      }
    }

    out.close();
  }
  else
    cout << "Unable to open file: out.txt" << endl;
  return;
}

void readIn(list<vector<RigTForm> >& keyframes){
  /*
    File format: Name.txt
    6         // keyframe count
    22        // RigTForm count per keyframe
    x y z     // translation
    w x y z   // rotation (quaternion)
    x y z
    w x y z
    .
    .
    .
  */
  
  string name;
  cout << "What is the file name: ";
  cin >> name; 

  ifstream in(name);
  if(in.is_open()){
     keyframes.clear();
     unsigned keyFrameCount, TFCount;
     in >> keyFrameCount >> TFCount;
     for(unsigned i=0; i<keyFrameCount ; i++){
       vector<RigTForm> frame;
       for(unsigned j=0; j<TFCount ; j++){
         double t_x, t_y, t_z;
         double r_w, r_x, r_y, r_z;
          
         in >> t_x >> t_y >> t_z;
         in >> r_w >> r_x >> r_y >> r_z;

         Cvec3 t(t_x, t_y, t_z);
         Quat q(r_w, r_x, r_y, r_z);

         RigTForm TF(t, q);
         frame.push_back(TF);
       }

       keyframes.push_back(frame);
    }
     

     in.close();
  }
  else
    cout << "Unable to open file: in.txt" << endl;

  return;
}

vector<RigTForm> getFromIndex(unsigned i){

  assert(i < keyframes.size());
  // requested index is guaranteed to be less than the list size
  unsigned indexCounter = 0;
  list<vector<RigTForm> >::const_iterator it=keyframes.begin();
  for( ; indexCounter<i; ++it)
    indexCounter++;

  return *it;

}

Quat cn(Quat q){
  if(q[0] < 0)
    q *= -1;

  return q;
}

void sLerp(int currI, int nextI, float alpha){
  
    // since pdf instructed us to keep the keyframes as
    // a list, we have to iterate every time to get the
    // array of keyframes in the corresponding indexes
    // instead of keeping them as arrays and reach them instantly
    vector<RigTForm> curr = getFromIndex(currI);
    vector<RigTForm> next = getFromIndex(nextI);

    rbtNodes.clear();
    dumpSgRbtNodes(g_world, rbtNodes);

    for(unsigned i=0; i<rbtNodes.size(); ++i){
      //RigTForm currentTForm = rbtNodes[i]->getRbt();
      RigTForm currentTForm = curr[i];

      // lerp translation
      const Cvec3 lerp = curr[i].getTranslation()*(1-alpha) + next[i].getTranslation()*alpha;
      currentTForm.setTranslation(lerp);

      //slerp rotations
      const Quat slerp = pow(cn( next[i].getRotation() * inv(curr[i].getRotation()) ), alpha)*curr[i].getRotation();
      currentTForm.setRotation(slerp);

      rbtNodes[i]->setRbt(currentTForm);
    }

    return;
}

bool interpolateAndDisplay(float t){
    float alpha = t - floor(t);

    if( floor(t)+1 < keyframes.size() ){
      
      // indexes of frames are passed to sLerp
      // which is lerp & slerp --> sLerp
      sLerp(floor(t), floor(t)+1, alpha);

      glutPostRedisplay();
      return false;
    }
    else
      return true;
}

static void animateTimerCallback(int ms){
  float t = (float)ms/(float)g_msBetweenKeyFrames;

  bool endReached = interpolateAndDisplay(t);
  if(!endReached){
     glutTimerFunc(1000/g_animateFramesPerSecond, animateTimerCallback,
                    ms + 1000/g_animateFramesPerSecond);
  }
  else{
    cout << "Animation ended" << endl;
    isPlaying = false;
  }
}

static void keyboard(const unsigned char key, const int x, const int y) {
	switch (key) {
		case 27:
			exit(0);             // ESC
		case 'h':
			cout << " ============== H E L P ==============\n\n"
			<< "h\t\thelp menu\n"
			<< "s\t\tsave screenshot\n"
			<< "f\t\tToggle flat shading on/off.\n"
			<< "o\t\tCycle object to edit\n"
			<< "v\t\tCycle view\n"
			<< "drag left mouse to rotate\n" << endl;
			break;
		case 's':
			glFlush();
			writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
			break;
		case 'f':
			g_activeShader ^= 1;
			break;
		case 'v':
			viewState++;
			viewState %= 3;
      if(viewState == 0)
        cout << " Eye is sky camera" << endl;
      if(viewState == 1)
        cout << " Eye is Red Robot" << endl;
      if(viewState == 2)
        cout << " Eye is Blue Robot" << endl;
			break;
		case 'o':
			manipState++;
			manipState %= 3;
      if(manipState == 0)
        cout << "Manipulating Sky Camera" << endl;
      if(manipState == 1)
        cout << "Manipulating Red Robot" << endl;
      if(manipState == 2)
        cout << "Manipulating Blue Robot" << endl;
			break;
		case 'm':
			worldSkyFrame = !worldSkyFrame;
      if(manipState == 0){
        if(worldSkyFrame)
          cout << "  World-Sky Frame" << endl;
        else
          cout << "  Sky-Sky Frame" << endl;
      }
			break;
    case ' ':
      if(keyframes.empty() == false){
        // copy current frame into scene graph
        for(unsigned i=0; i<rbtNodes.size(); ++i)
          rbtNodes[i]->setRbt((*currentFrame)[i]);
      }

      break;
    case 'u':
      {
        // get the scene graph
        rbtNodes.clear();
        dumpSgRbtNodes(g_world, rbtNodes);
        
        // copy scene graph into current key frame
        if(keyframes.empty() == false){

          for(unsigned i=0; i<rbtNodes.size(); ++i)
            (*currentFrame)[i] = rbtNodes[i]->getRbt();
          cout << "current frame is updated with the scene graph" << endl;
        }

        //if keyframes are empty
        else{
          cout << "keyframes are empty. This scene is pushed as the first keyframe" << endl;
          // container to push back to keyframe list
          vector<RigTForm> frame;

          // populate tehe container
          for(unsigned i=0; i<rbtNodes.size() ; i++)
            frame.push_back(rbtNodes[i]->getRbt()); 
          
          keyframes.push_back(frame);
          currentFrame = keyframes.begin();
        }
      }
      break;
    case '>':
      if(keyframes.empty() == false){
        list<vector<RigTForm> >::const_iterator lastFrame = keyframes.end();
        lastFrame--;
        
        if(currentFrame != lastFrame){
          currentFrame++;
          cout << "Getting current frame one forward > " << endl;

          // copy current frame into current state
          for(unsigned i=0; i<rbtNodes.size(); ++i)
              rbtNodes[i]->setRbt((*currentFrame)[i]);
        }
      }
      break;
    case '<':
      if(keyframes.empty() == false){
        if(currentFrame != keyframes.begin()){
          currentFrame--; 
          cout << "Getting current frame one back < " << endl;

          // copy current frame into current state
          for(unsigned i=0; i<rbtNodes.size(); ++i)
            rbtNodes[i]->setRbt((*currentFrame)[i]);
        }
      }
      break;
    case 'd':
      if(currentFrame != keyframes.end()){
        cout << "Deleting current frame" << endl;
        if(keyframes.size() == 1){
          keyframes.erase(currentFrame);
          currentFrame = keyframes.end();
        }
        else{
          if(currentFrame == keyframes.begin()){
            keyframes.erase(currentFrame);
            currentFrame = keyframes.begin();
          }
          else{
            list<vector<RigTForm> >::const_iterator toErase = currentFrame;
            currentFrame++;
            keyframes.erase(toErase);
          }
        }
      }
      else
        cout << "Unable to delete: Current frame is not defined!" << endl;
      break;
    case 'n':
      {
        cout << "New keyframe created from the current state" << endl;
        // clear Nodes and get the current state of the world
        rbtNodes.clear();
        dumpSgRbtNodes(g_world, rbtNodes);

        // container to push back to keyframe list
        vector<RigTForm> frame;

        // populate tehe container
        for(unsigned i=0; i<rbtNodes.size() ; i++)
          frame.push_back(rbtNodes[i]->getRbt());          
      

        if(currentFrame != keyframes.end()){
          list<vector<RigTForm> >::iterator whereTo = currentFrame;
          whereTo++;
          keyframes.insert(whereTo, frame);
          currentFrame++;
        }
        else{ //keyframe isn't defined
          // save current state into keyframes
          keyframes.push_back(frame);

          // assign current frame into the newly pushed frame
          currentFrame = keyframes.begin();
        }                                  
       
      }
      break;
      
    case 'i':
      readIn(keyframes);
      break;
    case 'w':
      writeOut(keyframes);
      break;

    case 'y':
      if(keyframes.size() < 4)
        cout << "WARNING: Not enough keyframes" << endl;
      else{
        if(isPlaying == false){
          cout << "Animation playing..." << endl;
          isPlaying = true;
          animateTimerCallback(0);
        }
        else{
          ; // stop the animation?
        }
      }
      break;
    case '+':
      if(g_msBetweenKeyFrames > 100)
        g_msBetweenKeyFrames -= 100;
      cout << (float)g_msBetweenKeyFrames/1000 << " seconds between frames" << endl;
      break;
    case '-':
      g_msBetweenKeyFrames += 100;
      cout << (float)g_msBetweenKeyFrames/1000 << " seconds between frames" << endl;
      break;

	}
	glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Assignment 4");                       // title the window

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
}

static void initGLState() {
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
}

static void constructRobot(shared_ptr<SgTransformNode> base, const Cvec3& color) {

  const double ARM_LEN = 0.7,
               ARM_THICK = 0.25,
               TORSO_LEN = 1.5,
               TORSO_THICK = 0.25,
               TORSO_WIDTH = 1,
               LEG_LEN = 0.9,
               LEG_WIDTH = 0.3,
               LEG_THICK = 0.3,
               HEAD_WIDTH = 0.6,
               HEAD_HEIGHT = 0.5,
               HEAD_THICK = 0.28;
  const int NUM_JOINTS = 10,
            NUM_SHAPES = 10;

  struct JointDesc {
    int parent;
    float x, y, z;
  };

  JointDesc jointDesc[NUM_JOINTS] = {
    {-1}, // torso
    {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0},  // upper right arm
    {1,  ARM_LEN, 0, 0                },  // lower right arm
    {0,  -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
    {3,  -ARM_LEN, 0, 0},                  // lower left arm
    {0,  TORSO_WIDTH/2, -TORSO_LEN/2, 0},  // upper right leg
    {5,  0, -LEG_LEN, 0                },   // lower right leg
    {0,  -TORSO_WIDTH/2, -TORSO_LEN/2, 0}, // upper left leg
    {7,  0, -LEG_LEN, 0                },  // lower left leg
    {0,  0, TORSO_LEN/4, 0}            // head
  };

  struct ShapeDesc {
    int parentJointId;
    float x, y, z, sx, sy, sz;
    shared_ptr<Geometry> geometry;
  };

  ShapeDesc shapeDesc[NUM_SHAPES] = {
    {0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
    {1, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // upper right arm
    {2, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower right arm
    {3, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // upper left arm
    {4, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower left arm   
    {5, LEG_WIDTH/2, 0, 0, LEG_WIDTH, LEG_LEN, LEG_THICK, g_cube}, // upper right leg
    {6, LEG_WIDTH/2, 0, 0, LEG_WIDTH, LEG_LEN, LEG_THICK, g_cube}, // lower right leg
    {7, -LEG_WIDTH/2, 0, 0, LEG_WIDTH, LEG_LEN, LEG_THICK, g_cube}, // upper left leg
    {8, -LEG_WIDTH/2, 0, 0, LEG_WIDTH, LEG_LEN, LEG_THICK, g_cube}, // lower left leg
    {9, 0, TORSO_LEN/2, 0, HEAD_WIDTH, HEAD_HEIGHT, HEAD_THICK, g_cube}
  };

  shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

  for (int i = 0; i < NUM_JOINTS; ++i) {
    if (jointDesc[i].parent == -1)
      jointNodes[i] = base;
    else {
      jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
      jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
    }
  }
  for (int i = 0; i < NUM_SHAPES; ++i) {
    shared_ptr<MyShapeNode> shape(
      new MyShapeNode(shapeDesc[i].geometry,
                      color,
                      Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                      Cvec3(0, 0, 0),
                      Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
    jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
  }
}

static void initScene() {
  g_world.reset(new SgRootNode());

  g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));

  g_groundNode.reset(new SgRbtNode());
  g_groundNode->addChild(shared_ptr<MyShapeNode>(
                           new MyShapeNode(g_ground, Cvec3(0.1, 0.95, 0.1))));

  g_robot1Node.reset(new SgRbtNode(RigTForm(Cvec3(-2, 1, 0))));
  g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 1, 0))));

  constructRobot(g_robot1Node, Cvec3(1, 0, 0)); // a Red robot
  constructRobot(g_robot2Node, Cvec3(0, 0, 1)); // a Blue robot

  g_world->addChild(g_skyNode);
  g_world->addChild(g_groundNode);
  g_world->addChild(g_robot1Node);
  g_world->addChild(g_robot2Node);
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    glewInit(); // load the OpenGL extensions

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.3") << endl;
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");

    initGLState();
    initShaders();
    initGeometry();
    initScene();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
