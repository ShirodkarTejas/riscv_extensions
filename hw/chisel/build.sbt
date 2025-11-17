// Build configuration for Sparse Attention Accelerator

name := "sattn-accelerator"
version := "1.0"
scalaVersion := "2.13.10"

libraryDependencies ++= Seq(
  "edu.berkeley.cs" %% "chisel3" % "3.6.0",
  "edu.berkeley.cs" %% "rocketchip" % "1.6.0"
)

scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-language:reflectiveCalls"
)

